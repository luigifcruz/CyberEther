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

    void update(Config config);

 private:
    void select(U64 index);

    Config config;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_OPTION_STEPPER_HH
