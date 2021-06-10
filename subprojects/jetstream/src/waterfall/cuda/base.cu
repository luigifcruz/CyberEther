#include "jetstream/waterfall/cuda.hpp"

namespace Jetstream::Waterfall {

CUDA::CUDA(Config& c) : Generic(c) {
    auto render = cfg.render;

    ymax = cfg.height;
    cudaMalloc(&out_dptr, in.buf.size() * ymax * sizeof(float));

    binTextureCfg.buffer = (uint8_t*)out_dptr;
    binTextureCfg.cudaInterop = true;
    JETSTREAM_ASSERT_SUCCESS(this->_initRender());
}

CUDA::~CUDA() {
    cudaFree(out_dptr);
}

Result CUDA::_compute() {
    cudaMemcpy(out_dptr+(inc*in.buf.size()), in.buf.data(), sizeof(float)*in.buf.size(), cudaMemcpyDeviceToDevice);

    return Result::SUCCESS;
}

Result CUDA::_present() {
    binTexture->fill();

    return Result::SUCCESS;
}

} // namespace Jetstream::Waterfall
