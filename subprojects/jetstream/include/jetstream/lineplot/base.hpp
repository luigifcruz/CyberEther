#ifndef JETSTREAM_LPT_BASE_H
#define JETSTREAM_LPT_BASE_H

#include "jetstream/lineplot/generic.hpp"
#include "jetstream/lineplot/cpu.hpp"

namespace Jetstream::Lineplot {

inline std::shared_ptr<CPU> Instantiate(Config& config, std::shared_ptr<Module> producer, CPU::I& input) {
    return std::make_shared<CPU>(config, producer, input);
}

} // namespace Jetstream::Lineplot

#endif
