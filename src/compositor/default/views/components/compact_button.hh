#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_COMPACT_BUTTON_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_COMPACT_BUTTON_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct CompactButton : public Sakura::Button {
    enum class Tone {
        Default,
        Primary,
        Strong,
        Action,
        Warning,
    };

    struct Config {
        std::string str;
        Tone tone = Tone::Default;
        bool disabled = false;
        Sakura::Callback<void()> onClick;
    };

    void update(Config config) {
        Sakura::Button::Config button{
            .str = std::move(config.str),
            .disabled = config.disabled,
            .padding = {14.0f, 5.0f},
            .onClick = std::move(config.onClick),
            .font = Sakura::Text::Font::SansBold,
        };
        switch (config.tone) {
            case Tone::Default:
                button.textScale = 0.9f;
                button.padding = {9.0f, 5.0f};
                button.font = Sakura::Text::Font::Sans;
                break;
            case Tone::Primary:
                button.colorKey = "primary_btn";
                button.hoveredColorKey = "primary_btn_hovered";
                button.activeColorKey = "primary_btn_hovered";
                button.borderColorKey = "primary_btn";
                button.textColorKey = "primary_btn_text";
                button.font = Sakura::Text::Font::Sans;
                break;
            case Tone::Strong:
                break;
            case Tone::Action:
                button.variant = Sakura::Button::Variant::Action;
                break;
            case Tone::Warning:
                button.variant = Sakura::Button::Variant::Action;
                button.colorKey = "warning_btn";
                button.hoveredColorKey = "warning_btn_hovered";
                button.activeColorKey = "warning_btn_active";
                button.borderColorKey = "warning_btn_outline";
                button.textColorKey = "warning_btn_text";
                break;
        }
        Button::update(std::move(button));
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_COMPACT_BUTTON_HH
