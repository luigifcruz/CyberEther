#include "jetstream/waterfall/cuda.hpp"

namespace Jetstream::Waterfall {

CUDA::CUDA(Config& c) : Generic(c) {
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto render = cfg.render;

    ymax = cfg.height;
    CUDA_CHECK_THROW(cudaMalloc(&out_dptr, in.buf.size() * ymax * sizeof(float)));

    binTextureCfg.buffer = (uint8_t*)out_dptr;
    binTextureCfg.cudaInterop = true;
    JETSTREAM_CHECK_THROW(this->_initRender());
}

CUDA::~CUDA() {
    cudaFree(out_dptr);
    cudaStreamDestroy(stream);
}

Result CUDA::_compute() {
    CUDA_CHECK(cudaMemcpyAsync(out_dptr+(inc*in.buf.size()), in.buf.data(), sizeof(float)*in.buf.size(),
            cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
