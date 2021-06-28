#ifndef JETSTREAM_LPT_CUDA_H
#define JETSTREAM_LPT_CUDA_H

#include "jetstream/lineplot/generic.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream {

class Lineplot::CUDA : public Lineplot  {
public:
    explicit CUDA(const Config &, Manifest &);
    ~CUDA();

protected:
    Result _compute();
    Result _present();

    size_t plot_len;
    float* plot_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream

#endif
