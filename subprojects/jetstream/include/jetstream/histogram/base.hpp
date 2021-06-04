#ifndef JETSTREAM_HST_BASE_H
#define JETSTREAM_HST_BASE_H

#include "jetstream/histogram/generic.hpp"
#include "jetstream/histogram/cpu.hpp"

namespace Jetstream::Histogram {

inline std::shared_ptr<Generic> Instantiate(Locale L, Config& config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<CPU>(config);
        default:
            JETSTREAM_ASSERT_SUCCESS(Result::ERROR);
    }
}

} // namespace Jetstream::Histogram

#endif
