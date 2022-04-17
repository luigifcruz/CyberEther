#ifndef JSTCORE_LINEPLOT_CUDA_H
#define JSTCORE_LINEPLOT_CUDA_H

#include "jstcore/lineplot/generic.hpp"

#include <cuda_runtime.h>

namespace Jetstream {

template<>
class Lineplot<Levice::CUDA> : public LineplotGeneric  {
public:
    explicit Lineplot(const Config&, const Input&);
    ~Lineplot();

protected:
    const Result underlyingCompute();

    size_t plot_len;
    float* plot_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream

#endif
