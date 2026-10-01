#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstring>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <filesystem>

// Forward-declare the plugin initializer to link against libnvinfer_plugin
extern "C" bool initLibNvInferPlugins(void* logger, const char* libNamespace);

#define CHECK_CUDA(status)                                                   \
    do {                                                                     \
        auto err = (status);                                                 \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1;                                                        \
        }                                                                    \
    } while (0)

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress verbose info, show warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

size_t getElementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kFP8: return 1;
        case nvinfer1::DataType::kBF16: return 2;
        case nvinfer1::DataType::kINT64: return 8;
#endif
        default: return 4;
    }
}

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::string s = "(";
    for (int i = 0; i < dims.nbDims; ++i) {
        s += std::to_string(dims.d[i]);
        if (i < dims.nbDims - 1) s += ", ";
    }
    s += ")";
    return s;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS] [ENGINE_PATH]\n"
              << "Options:\n"
              << "  -h, --help             Show this help message\n"
              << "  -w, --warmup <N>       Number of warmup iterations (default: 50)\n"
              << "  -n, --iterations <N>   Number of benchmark iterations (default: 500)\n"
              << "  -d, --device <ID>      CUDA Device ID (default: 0)\n"
              << "Positional:\n"
              << "  ENGINE_PATH            Path to .engine file (default: student_bev.engine)\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string engine_path = "student_bev.engine";
    int num_warmup = 50;
    int num_iterations = 500;
    int device_id = 0;

    // CLI Argument Parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            num_warmup = std::stoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) {
            num_iterations = std::stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg[0] != '-') {
            engine_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    CHECK_CUDA(cudaSetDevice(device_id));

    cudaDeviceProp device_prop;
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));

    std::cout << "======================================================================\n";
    std::cout << "               TensorRT C++ Latency Benchmarking Harness               \n";
    std::cout << "======================================================================\n";
    std::cout << " Engine Path      : " << engine_path << "\n";
    std::cout << " GPU Device       : " << device_prop.name << " (Device ID: " << device_id << ")\n";
    std::cout << " Compute Cap.     : " << device_prop.major << "." << device_prop.minor << "\n";
    std::cout << " Warmup Runs      : " << num_warmup << "\n";
    std::cout << " Benchmark Runs   : " << num_iterations << "\n";
    std::cout << " TensorRT Version : " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    // Initialize TensorRT Plugins
    if (!initLibNvInferPlugins(&gLogger, "")) {
        std::cerr << "[Warning] initLibNvInferPlugins returned false." << std::endl;
    }

    // Resolve Engine File path (support running from build directory or root)
    std::string resolved_path = engine_path;
    if (!std::filesystem::exists(resolved_path)) {
        std::vector<std::string> fallbacks = {
            "../../../../" + engine_path,
            "../../../" + engine_path,
            "../../" + engine_path,
            "../" + engine_path
        };
        for (const auto& candidate : fallbacks) {
            if (std::filesystem::exists(candidate)) {
                resolved_path = candidate;
                break;
            }
        }
    }

    if (!std::filesystem::exists(resolved_path)) {
        std::cerr << "[Error] Cannot find TensorRT engine file: " << engine_path << std::endl;
        std::cerr << "Please verify that the file exists or pass its path as the first argument." << std::endl;
        return 1;
    }

    std::ifstream file(resolved_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Error] Failed to open engine file: " << resolved_path << std::endl;
        return 1;
    }

    file.seekg(0, file.end);
    std::streamsize engine_size = file.tellg();
    if (engine_size <= 0) {
        std::cerr << "[Error] Engine file is empty or corrupted: " << resolved_path << std::endl;
        return 1;
    }
    file.seekg(0, file.beg);

    std::vector<char> engine_data(static_cast<size_t>(engine_size));
    file.read(engine_data.data(), engine_size);
    file.close();

    std::cout << "[Info] Engine file loaded from: " << resolved_path << " (" << (engine_size / (1024.0 * 1024.0)) << " MB)\n";

    // Create TensorRT Runtime, Engine, and Execution Context
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    if (!runtime) {
        std::cerr << "[Error] Failed to create IRuntime." << std::endl;
        return 1;
    }

    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(engine_data.data(), engine_size));
    if (!engine) {
        std::cerr << "[Error] Failed to deserialize CUDA engine." << std::endl;
        return 1;
    }

    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "[Error] Failed to create IExecutionContext." << std::endl;
        return 1;
    }

    // Configure known dynamic input dimensions if present
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims dims = context->getTensorShape(name);
            bool has_dynamic = false;
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] < 0) has_dynamic = true;
            }
            if (has_dynamic || std::string(name) == "camera_image") {
                if (std::string(name) == "camera_image") {
                    context->setInputShape(name, nvinfer1::Dims4{1, 3, 1280, 1920});
                } else if (std::string(name) == "intrinsics") {
                    context->setInputShape(name, nvinfer1::Dims3{1, 3, 3});
                } else if (std::string(name) == "extrinsics_inv") {
                    context->setInputShape(name, nvinfer1::Dims3{1, 4, 4});
                }
            }
        }
    }

    // Inspect IO Tensors and Allocate Device Memory
    std::cout << "[Info] Inspecting TensorRT Engine IO Bindings:\n";
    struct TensorBuffer {
        std::string name;
        nvinfer1::TensorIOMode io_mode;
        nvinfer1::DataType dtype;
        nvinfer1::Dims shape;
        size_t total_elements;
        size_t total_bytes;
        void* d_ptr = nullptr;
    };

    std::vector<TensorBuffer> buffers;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        TensorBuffer tb;
        tb.name = name;
        tb.io_mode = engine->getTensorIOMode(name);
        tb.dtype = engine->getTensorDataType(name);
        tb.shape = context->getTensorShape(name);

        size_t elements = 1;
        for (int d = 0; d < tb.shape.nbDims; ++d) {
            elements *= static_cast<size_t>(std::max<int64_t>(1, tb.shape.d[d]));
        }
        tb.total_elements = elements;
        tb.total_bytes = elements * getElementSize(tb.dtype);

        CHECK_CUDA(cudaMalloc(&tb.d_ptr, tb.total_bytes));

        if (tb.io_mode == nvinfer1::TensorIOMode::kINPUT) {
            // Fill input with non-zero dummy values (e.g. 0.5f)
            std::vector<float> dummy_host(elements, 0.5f);
            CHECK_CUDA(cudaMemcpy(tb.d_ptr, dummy_host.data(), tb.total_bytes, cudaMemcpyHostToDevice));
        } else {
            CHECK_CUDA(cudaMemset(tb.d_ptr, 0, tb.total_bytes));
        }

        // Set address using TensorRT 10/11 API
        if (!context->setTensorAddress(name, tb.d_ptr)) {
            std::cerr << "[Error] Failed to setTensorAddress for " << name << std::endl;
            return 1;
        }

        buffers.push_back(tb);

        std::cout << "  - [" << (tb.io_mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT " : "OUTPUT") << "] "
                  << std::setw(16) << std::left << tb.name
                  << " Shape: " << std::setw(20) << dimsToString(tb.shape)
                  << " Size: " << std::fixed << std::setprecision(2) << (tb.total_bytes / (1024.0 * 1024.0)) << " MB\n";
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup Phase
    std::cout << "\n[Info] Warming up CUDA execution pipeline (" << num_warmup << " runs)...\n";
    for (int i = 0; i < num_warmup; ++i) {
        if (!context->enqueueV3(stream)) {
            std::cerr << "[Error] enqueueV3 failed during warmup run " << i << std::endl;
            return 1;
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "[Info] Warmup completed successfully.\n";

    // Benchmarking Phase
    std::cout << "[Info] Benchmarking pure GPU inference (" << num_iterations << " runs)...\n";
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));

    std::vector<float> latencies_ms(num_iterations);

    for (int i = 0; i < num_iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start_event, stream));
        if (!context->enqueueV3(stream)) {
            std::cerr << "[Error] enqueueV3 failed during benchmark run " << i << std::endl;
            return 1;
        }
        CHECK_CUDA(cudaEventRecord(stop_event, stream));
        CHECK_CUDA(cudaEventSynchronize(stop_event));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_event, stop_event));
        latencies_ms[i] = ms;
    }

    // Statistical Computations
    std::vector<float> sorted_latencies = latencies_ms;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());

    double sum = std::accumulate(sorted_latencies.begin(), sorted_latencies.end(), 0.0);
    double mean = sum / num_iterations;

    double sq_diff_sum = 0.0;
    for (float val : sorted_latencies) {
        sq_diff_sum += (val - mean) * (val - mean);
    }
    double stddev = std::sqrt(sq_diff_sum / num_iterations);

    auto getPercentile = [&](double p) {
        size_t idx = static_cast<size_t>(std::ceil(p * num_iterations) - 1);
        idx = std::min(idx, sorted_latencies.size() - 1);
        return sorted_latencies[idx];
    };

    float min_lat = sorted_latencies.front();
    float p50_lat = getPercentile(0.50);
    float p90_lat = getPercentile(0.90);
    float p95_lat = getPercentile(0.95);
    float p99_lat = getPercentile(0.99);
    float max_lat = sorted_latencies.back();
    double fps = 1000.0 / mean;

    // Display Results
    std::cout << "\n======================================================================\n";
    std::cout << "                       BENCHMARK RESULTS SUMMARY                      \n";
    std::cout << "======================================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << " Metric                  Value (ms)           Throughput\n";
    std::cout << " --------------------------------------------------------------------\n";
    std::cout << " Mean Latency           : " << std::setw(8) << mean     << " ms           " << std::setprecision(1) << fps << " FPS\n";
    std::cout << std::setprecision(3);
    std::cout << " Std. Deviation         : " << std::setw(8) << stddev   << " ms\n";
    std::cout << " Min Latency            : " << std::setw(8) << min_lat  << " ms\n";
    std::cout << " Median (P50)           : " << std::setw(8) << p50_lat  << " ms\n";
    std::cout << " 90th Percentile (P90)  : " << std::setw(8) << p90_lat  << " ms\n";
    std::cout << " 95th Percentile (P95)  : " << std::setw(8) << p95_lat  << " ms\n";
    std::cout << " 99th Percentile (P99)  : " << std::setw(8) << p99_lat  << " ms\n";
    std::cout << " Max Latency            : " << std::setw(8) << max_lat  << " ms\n";
    std::cout << "======================================================================\n";

    // Autonomous Driving Real-Time Constraints Assessment
    std::cout << "\n Autonomous Driving Viability Assessment:\n";
    if (mean <= 33.33) {
        std::cout << "  [PASS] Real-time capable for 30 Hz sensor streams (" << (1000.0 / 30.0) << " ms budget).\n";
    } else {
        std::cout << "  [WARN] Exceeds 30 Hz budget (" << (1000.0 / 30.0) << " ms).\n";
    }
    if (mean <= 100.0) {
        std::cout << "  [PASS] Real-time capable for 10 Hz LiDAR/Camera sweeps (" << (1000.0 / 10.0) << " ms budget).\n";
    } else {
        std::cout << "  [FAIL] Does not meet 10 Hz real-time perception requirement.\n";
    }
    std::cout << "======================================================================\n\n";

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaStreamDestroy(stream));

    for (auto& tb : buffers) {
        if (tb.d_ptr) {
            cudaFree(tb.d_ptr);
        }
    }

    return 0;
}
