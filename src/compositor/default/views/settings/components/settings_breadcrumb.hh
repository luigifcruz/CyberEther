#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_BREADCRUMB_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_BREADCRUMB_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct SettingsBreadcrumb : public Sakura::Stack {
    struct Config {
        std::string root;
        std::string current;
        std::function<void()> onRoot;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        rootButton.update({
            .str = std::move(config.root),
            .variant = Sakura::Button::Variant::Text,
            .textColorKey = "text_secondary",
            .onClick = std::move(config.onRoot),
            .font = Text::Font::Sans,
        });
        chevron.update({
            .str = ICON_FA_CHEVRON_RIGHT,
            .tone = Text::Tone::Secondary,
            .scale = 0.7f,
            .verticalOffset = 1.0f,
        });
        current.update({
            .str = std::move(config.current),
            .font = Text::Font::Sans,
        });
        trail.update({
            .axis = Axis::Horizontal,
            .spacing = 10.0f,
            .items = {
                {.child = &rootButton},
                {.child = &chevron},
                {.child = &current},
            },
        });
        spacing.update({.lines = 3});
        Stack::update({
            .spacing = 4.0f,
            .items = {
                {.child = &trail},
                {.child = &spacing},
            },
        });
    }

 private:
    Sakura::Button rootButton;
    Sakura::Text chevron;
    Sakura::Text current;
    Sakura::Stack trail;
    Sakura::Spacer spacing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_BREADCRUMB_HH
