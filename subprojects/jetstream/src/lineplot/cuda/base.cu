#include "jetstream/lineplot/cuda.hpp"

namespace Jetstream::Lineplot {

CUDA::CUDA(Config& c) : Generic(c) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    plot_len = plot.size() * sizeof(plot[0]);
    cudaMalloc(&plot_dptr, plot_len);
    cudaMemcpy(plot_dptr, plot.data(), plot_len, cudaMemcpyHostToDevice);

    plotVbo.data = plot_dptr;
    plotVbo.cudaInterop = true;
    JETSTREAM_ASSERT_SUCCESS(this->_initRender());
}

CUDA::~CUDA() {
    cudaFree(plot_dptr);
    cudaStreamDestroy(stream);
}

Result CUDA::_compute() {
    size_t elementSize = sizeof(float);
    size_t srcPitchInBytes = 1 * elementSize;
    size_t dstPitchInBytes = 3 * elementSize;
    size_t width = 1 * elementSize;
    size_t height = in.buf.size();

    cudaMemcpy2DAsync(plot_dptr + 1, dstPitchInBytes, in.buf.data(), srcPitchInBytes,
        width, height, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return Result::SUCCESS;
}

Result CUDA::_present() {
    lineVertex->update();

    return Result::SUCCESS;
}

} // namespace Jetstream::Lineplot
