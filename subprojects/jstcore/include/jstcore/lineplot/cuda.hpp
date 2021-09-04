#ifndef JSTCORE_LINEPLOT_CUDA_H
#define JSTCORE_LINEPLOT_CUDA_H

#include "jstcore/lineplot/generic.hpp"

#include <cuda_runtime.h>

namespace Jetstream::Lineplot {

class CUDA : public Generic  {
public:
    explicit CUDA(const Config &, const Input &);
    ~CUDA();

protected:
    Result underlyingCompute() final;

    size_t plot_len;
    float* plot_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream::Lineplot

#endif
