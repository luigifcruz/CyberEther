#ifndef JETSTREAM_LPT_BASE_H
#define JETSTREAM_LPT_BASE_H

#include "jetstream/lineplot/generic.hpp"
#include "jetstream/lineplot/cpu.hpp"

namespace Jetstream::Lineplot {

inline std::shared_ptr<Generic> Instantiate(Locale L, Config& config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<CPU>(config);
        default:
            JETSTREAM_ASSERT_SUCCESS(Result::ERROR);
    }
}

} // namespace Jetstream::Lineplot

#endif
