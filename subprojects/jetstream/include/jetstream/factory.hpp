#ifndef JETSTREAM_FACTORY_H
#define JETSTREAM_FACTORY_H

#include "jetstream/type.hpp"
#include "jetstream/scheduler/base.hpp"

namespace Jetstream {

inline std::shared_ptr<Scheduler> Factory(const Launch & L, const std::shared_ptr<Module> & m,
        const Dependencies & d) {
    switch (L) {
        case Jetstream::Launch::ASYNC:
            return std::make_shared<Async>(m, d);
        case Jetstream::Launch::SYNC:
            return std::make_shared<Sync>(m, d);
    }
}

template<typename T>
inline std::shared_ptr<T> Factory(const Locale & L, const typename T::Config & config, Manifest & inputs) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<typename T::CPU>(config, inputs);
        case Jetstream::Locale::CUDA:
            return std::make_shared<typename T::CUDA>(config, inputs);
    }
}

} // namespace Jetstream

#endif
