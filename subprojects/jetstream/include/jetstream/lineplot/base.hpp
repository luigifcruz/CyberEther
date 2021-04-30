#ifndef JETSTREAM_LPT_BASE_H
#define JETSTREAM_LPT_BASE_H

#include "jetstream/lineplot/generic.hpp"
#include "jetstream/lineplot/cpu.hpp"

namespace Jetstream::Lineplot {

inline std::shared_ptr<CPU> Instantiate(Config& c, DF::CPU::CF32V& d) {
    return std::make_shared<CPU>(c, d);
}

} // namespace Jetstream::Lineplot

#endif
