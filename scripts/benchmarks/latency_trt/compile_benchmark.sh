#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building TensorRT Latency Benchmark ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake ..
cmake --build . --config Release -j$(nproc)

echo ""
echo "=== Build Successful! ==="
echo "Binary created at: ${BUILD_DIR}/benchmark_latency"
echo "Example usage:"
echo "  # From project root:"
echo "  ./scripts/benchmarks/latency_trt/build/benchmark_latency student_bev.engine -w 50 -n 500"
echo "  # Or from build directory:"
echo "  cd ${BUILD_DIR} && ./benchmark_latency ../../../../student_bev.engine -w 50 -n 500"

