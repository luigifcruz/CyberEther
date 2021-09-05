#include "jstcore/waterfall/cuda.hpp"

namespace Jetstream::Waterfall {

CUDA::CUDA(const Config & config, const Input & input) : Generic(config, input) {
    ymax = config.size.height;
    JST_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    JST_CUDA_CHECK_THROW(cudaMalloc(&out_dptr, input.in.buf.size() * ymax * sizeof(float)));
    JST_CHECK_THROW(this->initRender((uint8_t*)out_dptr, config.render->cudaInteropSupported()));
}

CUDA::~CUDA() {
    cudaFree(out_dptr);
    cudaStreamDestroy(stream);
}

Result CUDA::underlyingCompute() {
    JST_CUDA_CHECK(cudaMemcpyAsync(out_dptr+(inc*input.in.buf.size()), input.in.buf.data(),
            sizeof(float)*input.in.buf.size(), cudaMemcpyDeviceToDevice, stream));
    JST_CUDA_CHECK(cudaStreamSynchronize(stream));
    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
