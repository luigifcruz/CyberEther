#ifndef JETSTREAM_WTF_BASE_H
#define JETSTREAM_WTF_BASE_H

#include "jetstream/waterfall/generic.hpp"
#include "jetstream/waterfall/cpu.hpp"

namespace Jetstream::Waterfall {

inline std::shared_ptr<Generic> Instantiate(Locale L, Config& config) {
    switch (L) {
        case Jetstream::Locale::CPU:
            return std::make_shared<CPU>(config);
        default:
            JETSTREAM_ASSERT_SUCCESS(Result::ERROR);
    }
}

} // namespace Jetstream::Waterfall

#endif
