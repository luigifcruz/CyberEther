#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_HEADING_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_HEADING_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct SidebarHeading : public Sakura::Stack {
    static constexpr F32 Inset = 9.0f;

    struct Config {
        std::string label;
        bool title = false;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        inset.update({});
        text.update({
            .str = std::move(config.label),
            .font = config.title ? Text::Font::DisplaySemiBold : Text::Font::Display,
            .tone = config.title ? Text::Tone::Primary : Text::Tone::Secondary,
            .scale = config.title ? 1.8f : Sakura::Typography::Scale::Title,
        });
        Stack::update({
            .axis = Axis::Horizontal,
            .spacing = 0.0f,
            .items = {
                {.child = &inset, .size = Inset},
                {.child = &text},
            },
        });
    }

 private:
    Sakura::Spacer inset;
    Sakura::Text text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_HEADING_HH
