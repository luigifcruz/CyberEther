#include "jstcore/lineplot/cuda.hpp"

namespace Jetstream::Lineplot {

CUDA::CUDA(const Config & config, const Input & input) : Generic(config, input) {
    plot_len = plot.size() * sizeof(plot[0]);
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK_THROW(cudaMalloc(&plot_dptr, plot_len));
    CUDA_CHECK_THROW(cudaMemcpy(plot_dptr, plot.data(), plot_len, cudaMemcpyHostToDevice));
    JETSTREAM_CHECK_THROW(this->initRender(plot_dptr, config.render->cudaInteropSupported()));
}

CUDA::~CUDA() {
    cudaFree(plot_dptr);
    cudaStreamDestroy(stream);
}

Result CUDA::underlyingCompute() {
    size_t elementSize = sizeof(float);
    size_t srcPitchInBytes = 1 * elementSize;
    size_t dstPitchInBytes = 3 * elementSize;
    size_t width = 1 * elementSize;
    size_t height = input.in.buf.size();

    CUDA_CHECK(cudaMemcpy2DAsync(plot_dptr + 1, dstPitchInBytes, in.buf.data(), srcPitchInBytes,
        width, height, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
