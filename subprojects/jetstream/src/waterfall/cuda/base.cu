#include "jetstream/waterfall/cuda.hpp"

namespace Jetstream {
namespace Waterfall {
    
CUDA::CUDA(const Config& c) : Generic(c) {
    ymax = cfg.size.height;
    CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK_THROW(cudaMalloc(&out_dptr, in.buf.size() * ymax * sizeof(float)));
    JETSTREAM_CHECK_THROW(this->_initRender((uint8_t*)out_dptr, cfg.render->cudaInteropSupported()));
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

} // namespace Waterfall
} // namespace Jetstream
