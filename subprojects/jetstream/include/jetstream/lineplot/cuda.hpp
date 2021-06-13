#ifndef JETSTREAM_LPT_CUDA_H
#define JETSTREAM_LPT_CUDA_H

#include "jetstream/lineplot/generic.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

namespace Jetstream::Lineplot {

class CUDA : public Generic  {
public:
    explicit CUDA(Config&);
    ~CUDA();

protected:
    Result _compute();
    Result _present();

    size_t plot_len;
    float* plot_dptr;

    cudaStream_t stream;
};

} // namespace Jetstream::Lineplot

#endif
