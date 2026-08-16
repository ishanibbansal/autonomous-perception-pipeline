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

// TensorRT Logger (Required to instantiate the runtime)
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
        // 1. Initialize Sub/Pub
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, std::bind(&StudentBEVNode::imageCallback, this, std::placeholders::_1));
        
        heatmap_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/bev_heatmap", 10);

        // 2. Load the TensorRT Engine
        loadEngine("/home/ishan/autonomous-perception-pipeline/student_bev.engine");

        // 3. Allocate CUDA Device Memory for inputs/outputs
        cudaMalloc(&d_camera_image_, 1 * 3 * 1280 * 1920 * sizeof(float));
        cudaMalloc(&d_intrinsics_, 1 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_extrinsics_inv_, 1 * 4 * 4 * sizeof(float));
        cudaMalloc(&d_bev_occupancy_, 1 * 160 * 160 * sizeof(float)); // Logits output

        // 4. Set Tensor Addresses for TRT 11 execution context
        context_->setTensorAddress("camera_image", d_camera_image_);
        context_->setTensorAddress("intrinsics", d_intrinsics_);
        context_->setTensorAddress("extrinsics_inv", d_extrinsics_inv_);
        context_->setTensorAddress("bev_occupancy", d_bev_occupancy_);

        cudaStreamCreate(&stream_);
        
        RCLCPP_INFO(this->get_logger(), "TensorRT Engine Initialized and Ready for Inference.");
    }

    ~StudentBEVNode() {
        cudaStreamDestroy(stream_);
        cudaFree(d_camera_image_);
        cudaFree(d_intrinsics_);
        cudaFree(d_extrinsics_inv_);
        cudaFree(d_bev_occupancy_);
    }

private:
    void loadEngine(const std::string& engine_path) {
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
        
        // Define runtime input shapes (Batch size = 1)
        context_->setInputShape("camera_image", nvinfer1::Dims4{1, 3, 1280, 1920});
        context_->setInputShape("intrinsics", nvinfer1::Dims3{1, 3, 3});
        context_->setInputShape("extrinsics_inv", nvinfer1::Dims3{1, 4, 4});
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // 1. Preprocess OpenCV Image (BGR to RGB, scale to [0, 1], HWC to CHW)
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);

            // Simple contiguous memory copy mapping for CHW format
            std::vector<float> chw_image(3 * 1280 * 1920);
            std::vector<cv::Mat> channels(3);
            cv::split(frame, channels);
            size_t channel_size = 1280 * 1920 * sizeof(float);
            memcpy(chw_image.data(), channels[0].data, channel_size);
            memcpy(chw_image.data() + 1280 * 1920, channels[1].data, channel_size);
            memcpy(chw_image.data() + 2 * 1280 * 1920, channels[2].data, channel_size);

            // 2. Define static calibration matrices (Replace with your actual Waymo values later)
            float intrinsics[9] = {1.0, 0.0, 960.0, 0.0, 1.0, 640.0, 0.0, 0.0, 1.0};
            float extrinsics_inv[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

            // 3. Host-to-Device Memory Transfers
            cudaMemcpyAsync(d_camera_image_, chw_image.data(), chw_image.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(d_intrinsics_, intrinsics, 9 * sizeof(float), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(d_extrinsics_inv_, extrinsics_inv, 16 * sizeof(float), cudaMemcpyHostToDevice, stream_);

            // 4. Execute TRT 11 Inference
            context_->enqueueV3(stream_);

            // 5. Device-to-Host Transfer for Output
            std::vector<float> output_logits(160 * 160);
            cudaMemcpyAsync(output_logits.data(), d_bev_occupancy_, output_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_); // Wait for GPU to finish

            // 6. Post-Process Output (Sigmoid) & Publish as Image
            cv::Mat heatmap(160, 160, CV_8UC1);
            for (int i = 0; i < 160 * 160; ++i) {
                float prob = 1.0f / (1.0f + std::exp(-output_logits[i])); // Sigmoid
                heatmap.data[i] = static_cast<uint8_t>(prob * 255.0f);
            }

            // Apply a nice colormap (like 'magma' from matplotlib) and publish
            cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_MAGMA);
            auto heatmap_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", heatmap).toImageMsg();
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
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StudentBEVNode>());
    rclcpp::shutdown();
    return 0;
}