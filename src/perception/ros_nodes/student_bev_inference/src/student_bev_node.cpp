#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

// Forward-declare the plugin initializer to bypass the missing NvInferPlugin.h
extern "C" bool initLibNvInferPlugins(void* logger, const char* libNamespace);

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

class StudentBEVNode : public rclcpp::Node {
public:
    StudentBEVNode() : Node("student_bev_node") {
        this->declare_parameter<std::string>("engine_path", "student_bev.engine");
        std::string engine_path = this->get_parameter("engine_path").as_string();

        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, std::bind(&StudentBEVNode::imageCallback, this, std::placeholders::_1));
        
        heatmap_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/bev_heatmap", 10);

        cudaMalloc(&d_camera_image_, 1 * 3 * 1280 * 1920 * sizeof(float));
        cudaMalloc(&d_intrinsics_, 1 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_extrinsics_inv_, 1 * 4 * 4 * sizeof(float));
        cudaMalloc(&d_bev_occupancy_, 1 * 160 * 160 * sizeof(float));
        cudaMalloc(&d_dummy_output_, 1 * 160 * 160 * sizeof(float)); 

        loadEngine(engine_path);

        context_->setTensorAddress("camera_image", d_camera_image_);
        context_->setTensorAddress("intrinsics", d_intrinsics_);
        context_->setTensorAddress("extrinsics_inv", d_extrinsics_inv_);

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                if (std::string(name) == "bev_occupancy") {
                    context_->setTensorAddress(name, d_bev_occupancy_);
                    RCLCPP_INFO(this->get_logger(), "Bound primary output memory to: %s", name);
                } else {
                    context_->setTensorAddress(name, d_dummy_output_);
                    RCLCPP_INFO(this->get_logger(), "Bound extra dummy output to: %s", name);
                }
            }
        }

        cudaStreamCreate(&stream_);
        RCLCPP_INFO(this->get_logger(), "TensorRT Engine Initialized and Ready for Inference.");
    }

    ~StudentBEVNode() {
        cudaStreamDestroy(stream_);
        cudaFree(d_camera_image_);
        cudaFree(d_intrinsics_);
        cudaFree(d_extrinsics_inv_);
        cudaFree(d_bev_occupancy_);
        cudaFree(d_dummy_output_);
    }

private:
    void loadEngine(const std::string& engine_path) {
        initLibNvInferPlugins(&gLogger, "");

        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open engine file: %s", engine_path.c_str());
            return;
        }
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), size));
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        
        context_->setInputShape("camera_image", nvinfer1::Dims4{1, 3, 1280, 1920});
        context_->setInputShape("intrinsics", nvinfer1::Dims3{1, 3, 3});
        context_->setInputShape("extrinsics_inv", nvinfer1::Dims3{1, 4, 4});
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // 1. Save a pristine copy of the raw BGR frame for the side-by-side visualization
            cv::Mat orig_frame = cv_bridge::toCvCopy(msg, "bgr8")->image.clone();

            // 2. Preprocess the inference frame
            cv::Mat frame = orig_frame.clone();
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);

            // ImageNet normalization REMOVED to perfectly match the PyTorch WaymoDataset

            std::vector<float> chw_image(3 * 1280 * 1920);
            std::vector<cv::Mat> channels(3);
            cv::split(frame, channels);
            size_t channel_size = 1280 * 1920 * sizeof(float);
            memcpy(chw_image.data(), channels[0].data, channel_size);
            memcpy(chw_image.data() + 1280 * 1920, channels[1].data, channel_size);
            memcpy(chw_image.data() + 2 * 1280 * 1920, channels[2].data, channel_size);

            float intrinsics[9] = {
                2083.091212f, 0.0f, 957.293829f,
                0.0f, 2083.091212f, 650.569793f,
                0.0f, 0.0f, 1.0f
            };

            float extrinsics_inv[16] = {
                 0.0f,  0.0f,  1.0f,  2.0f,
                -1.0f,  0.0f,  0.0f,  0.0f,
                 0.0f, -1.0f,  0.0f,  1.5f,
                 0.0f,  0.0f,  0.0f,  1.0f
            };

            cudaMemcpyAsync(d_camera_image_, chw_image.data(), chw_image.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(d_intrinsics_, intrinsics, 9 * sizeof(float), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(d_extrinsics_inv_, extrinsics_inv, 16 * sizeof(float), cudaMemcpyHostToDevice, stream_);

            context_->enqueueV3(stream_);

            std::vector<float> output_logits(160 * 160);
            cudaMemcpyAsync(output_logits.data(), d_bev_occupancy_, output_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_); 

            // 3. Post-Process the Heatmap
            cv::Mat heatmap(160, 160, CV_8UC1);
            for (int i = 0; i < 160 * 160; ++i) {
                float prob = 1.0f / (1.0f + std::exp(-output_logits[i]));
                heatmap.data[i] = static_cast<uint8_t>(prob * 255.0f);
            }

            // CRITICAL FIX: Flip vertically to correct OpenCV top-down rendering
            cv::flip(heatmap, heatmap, 0);
            cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_MAGMA);

            // 4. Create the Side-by-Side Visualization
            cv::Mat display_img, display_heatmap, combined;
            
            // Resize original image down to 960x640
            cv::resize(orig_frame, display_img, cv::Size(960, 640));
            
            // Scale heatmap up to 640x640 to match the height
            cv::resize(heatmap, display_heatmap, cv::Size(640, 640));
            
            // Stitch them horizontally
            cv::hconcat(display_img, display_heatmap, combined);

            // Publish the combined image
            auto heatmap_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", combined).toImageMsg();
            heatmap_pub_->publish(*heatmap_msg);

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr heatmap_pub_;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_;
    void* d_camera_image_ = nullptr;
    void* d_intrinsics_ = nullptr;
    void* d_extrinsics_inv_ = nullptr;
    void* d_bev_occupancy_ = nullptr;
    void* d_dummy_output_ = nullptr;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StudentBEVNode>());
    rclcpp::shutdown();
    return 0;
}