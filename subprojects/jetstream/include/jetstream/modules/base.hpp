#ifndef JETSTREAM_MODULE_BASE_H
#define JETSTREAM_MODULE_BASE_H

#include "jetstream/modules/fft/base.hpp"
#include "jetstream/modules/lineplot/base.hpp"
#include "jetstream/modules/waterfall/base.hpp"

namespace Jetstream {

template<typename T>
inline std::shared_ptr<T> Factory(const Locale & L, const typename T::Config & config, IO & inputs) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<typename T::CPU>(config, inputs);
        case Jetstream::Locale::CUDA:
            return std::make_shared<typename T::CUDA>(config, inputs);
    }
}

} // namespace Jetstream

#endif
