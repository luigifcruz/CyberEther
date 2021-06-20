#ifndef JETSTREAM_WTF_CUDA_H
#define JETSTREAM_WTF_CUDA_H

#include "jetstream/waterfall/generic.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream::Waterfall {

class CUDA : public Generic  {
public:
    explicit CUDA(const Config &);
    ~CUDA();

protected:
    Result _compute();

    float* out_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream::Waterfall

#endif
