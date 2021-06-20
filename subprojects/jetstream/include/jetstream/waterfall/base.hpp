#ifndef JETSTREAM_WTF_BASE_H
#define JETSTREAM_WTF_BASE_H

#include "jetstream/waterfall/generic.hpp"
#include "jetstream/waterfall/cpu.hpp"
#ifdef JETSTREAM_WTF_CUDA_AVAILABLE
#include "jetstream/waterfall/cuda.hpp"
#endif

namespace Jetstream::Waterfall {

inline std::shared_ptr<Generic> Instantiate(Locale L, const Config& config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<CPU>(config);
#ifdef JETSTREAM_WTF_CUDA_AVAILABLE
        case Jetstream::Locale::CUDA:
            return std::make_shared<CUDA>(config);
#endif
        default:
            JETSTREAM_CHECK_THROW(Jetstream::Result::ERROR);
    }
}

} // namespace Jetstream::Waterfall

#endif
