#ifndef JSTCORE_WATERFALL_CUDA_H
#define JSTCORE_WATERFALL_CUDA_H

#include "jstcore/waterfall/generic.hpp"

#include <cuda_runtime.h>

namespace Jetstream::Waterfall {

class CUDA : public Generic  {
public:
    explicit CUDA(const Config &, const Input &);
    ~CUDA();

protected:
    Result underlyingCompute() final;

    float* out_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream::Waterfall

#endif

