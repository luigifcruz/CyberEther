#ifndef JSTCORE_LINEPLOT_CUDA_H
#define JSTCORE_LINEPLOT_CUDA_H

#include "jstcore/lineplot/generic.hpp"

#include <cuda_runtime.h>

namespace Jetstream::Lineplot {

/**
 * This is the implementation of the Lineplot using CUDA.
 * It will make use of graphics interopability with CUDA.
 * The choosen render backend has to be compatible with it.
 * Otherwise, just use the CPU implementation of Lineplot.
 */
class CUDA : public Generic  {
public:
    explicit CUDA(const Config&, const Input&);
    ~CUDA();

protected:
    Result underlyingCompute() final;

    size_t plot_len;
    float* plot_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream::Lineplot

#endif
