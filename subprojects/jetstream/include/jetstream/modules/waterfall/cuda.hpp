#ifndef JETSTREAM_WTF_CUDA_H
#define JETSTREAM_WTF_CUDA_H

#include "jetstream/modules/waterfall/generic.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream {

class Waterfall::CUDA : public Waterfall  {
public:
    explicit CUDA(const Config & cfg, Connections & input);
    ~CUDA();

protected:
    Result _compute();

    float* out_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream

#endif
