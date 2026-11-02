#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_OPTION_STEPPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_OPTION_STEPPER_HH

#include "jetstream/render/sakura/base.hh"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct OptionStepper : public Sakura::PillStepper {
    struct Config {
        std::vector<std::string> options;
        U64 selected = 0;
        U64 fallback = 0;
        F32 width = 132.0f;
        Sakura::Callback<void(U64)> onSelect;
    };

    void update(Config config) {
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

 private:
    void select(U64 index) {
        if (index < config.options.size() && config.onSelect) {
            config.onSelect(index);
        }
    }

    Config config;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_OPTION_STEPPER_HH
