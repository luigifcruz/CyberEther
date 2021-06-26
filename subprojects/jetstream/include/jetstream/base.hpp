#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "jetstream/engine.hpp"
#include "jetstream/fft/base.hpp"

namespace Jetstream {

template<typename T>
inline std::shared_ptr<T> Factory(const Locale & L, const typename T::Config & config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<typename T::CPU>(config);
        case Jetstream::Locale::CUDA:
            return std::make_shared<typename T::CUDA>(config);
    }
}

} // namespace Jetstream

#endif
