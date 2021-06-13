#include "jetstream/waterfall/cuda.hpp"

namespace Jetstream::Waterfall {

CUDA::CUDA(Config& c) : Generic(c) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    auto render = cfg.render;

    ymax = cfg.height;
    cudaMalloc(&out_dptr, in.buf.size() * ymax * sizeof(float));

    binTextureCfg.buffer = (uint8_t*)out_dptr;
    binTextureCfg.cudaInterop = true;
    JETSTREAM_ASSERT_SUCCESS(this->_initRender());
}

CUDA::~CUDA() {
    cudaFree(out_dptr);
    cudaStreamDestroy(stream);
}

Result CUDA::_compute() {
    cudaMemcpyAsync(out_dptr+(inc*in.buf.size()), in.buf.data(), sizeof(float)*in.buf.size(),
            cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return Result::SUCCESS;
}

Result CUDA::_present() {
    binTexture->fill();

    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
