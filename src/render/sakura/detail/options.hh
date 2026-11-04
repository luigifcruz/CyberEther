#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_OPTIONS_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_OPTIONS_HH

#include <jetstream/types.hh>

#include <functional>
#include <optional>

namespace Jetstream::Sakura {

template<typename Config>
std::optional<U64> SelectedOption(const Config& config) {
    if (config.onSelect) {
        return config.selectedIndex;
    }
    for (U64 index = 0; index < config.options.size(); ++index) {
        if (config.options[index] == config.value) {
            return index;
        }
    }
    return std::nullopt;
}

template<typename Config>
std::function<void(U64)> OptionSelected(const Config& config) {
    return [&config](U64 index) {
        if (index >= config.options.size()) {
            return;
        }
        const auto option = config.options[index];
        if (config.onSelect) {
            config.onSelect(index);
        }
        if (config.onChange) {
            config.onChange(option);
        }
    };
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_OPTIONS_HH
