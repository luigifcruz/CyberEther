#include "jstcore/lineplot/cuda.hpp"

namespace Jetstream::Lineplot {

CUDA::CUDA(const Config & config, const Input & input) : Generic(config, input) {
    if (!config.render->cudaInteropSupported()) {
        std::cerr << "[LINEPLOT::CUDA] This implementation expects the choosen render to be \
            compatible with CUDA interopability. Please use the standard CPU implementation." << std::endl;
        JST_CHECK_THROW(Result::ERROR);
    }

    if ((input.in.location & Locale::CUDA) != Locale::CUDA) {
        std::cerr << "[LINEPLOT::CUDA] This implementation expects a Locale::CUDA input." << std::endl;
        JST_CHECK_THROW(Result::ERROR);
    }

    plot_len = plot.size() * sizeof(plot[0]);
    JST_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    JST_CUDA_CHECK_THROW(cudaMalloc(&plot_dptr, plot_len));
    JST_CUDA_CHECK_THROW(cudaMemcpy(plot_dptr, plot.data(), plot_len, cudaMemcpyHostToDevice));
    JST_CHECK_THROW(this->initRender(plot_dptr, true));
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

    JST_CUDA_CHECK(cudaMemcpy2DAsync(plot_dptr + 1, dstPitchInBytes, input.in.buf.data(), srcPitchInBytes,
        width, height, cudaMemcpyDeviceToDevice, stream));
    JST_CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
