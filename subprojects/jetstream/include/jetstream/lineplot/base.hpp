#ifndef JETSTREAM_LPT_BASE_H
#define JETSTREAM_LPT_BASE_H

#include "jetstream/lineplot/generic.hpp"
#include "jetstream/lineplot/cpu.hpp"
#ifdef JETSTREAM_LPT_CUDA_AVAILABLE
#include "jetstream/lineplot/cuda.hpp"
#endif

namespace Jetstream::Lineplot {

inline std::shared_ptr<Generic> Instantiate(Locale L, Config& config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<CPU>(config);
#ifdef JETSTREAM_LPT_CUDA_AVAILABLE
        case Jetstream::Locale::CUDA:
            return std::make_shared<CUDA>(config);
#endif
        default:
            JETSTREAM_CHECK_THROW(Jetstream::Result::ERROR);
    }
}

} // namespace Jetstream::Lineplot

#endif
