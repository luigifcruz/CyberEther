#include "option_stepper.hh"

namespace Jetstream {

void OptionStepper::update(Config config) {
    this->config = std::move(config);
    const U64 count = this->config.options.size();
    const U64 index = count == 0 ? 0 : std::min<U64>(this->config.selected, count - 1);
    PillStepper::update({
        .label = count == 0 ? std::string() : this->config.options[index],
        .canDecrement = index > 0,
        .canIncrement = index + 1 < count,
        .size = {this->config.width, 34.0f},
        .onDecrement = [this, index]() {
            if (index > 0) {
                select(index - 1);
            }
        },
        .onIncrement = [this, index, count]() {
            if (index + 1 < count) {
                select(index + 1);
            }
        },
        .onReset = [this]() {
            select(this->config.fallback);
        },
    });
}

void OptionStepper::select(U64 index) {
    if (index < config.options.size() && config.onSelect) {
        config.onSelect(index);
    }
}

}  // namespace Jetstream
