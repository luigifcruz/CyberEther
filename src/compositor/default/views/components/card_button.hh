#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CARD_BUTTON_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CARD_BUTTON_HH

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct CardButton : public Sakura::Button {
    static constexpr F32 PillRounding = 1000.0f;

    enum class Tone {
        Card,
        Primary,
        Action,
    };

    struct Config {
        std::string str;
        Tone tone = Tone::Card;
        F32 width = Sakura::Fill;
        F32 height = 40.0f;
        F32 textScale = 1.0f;
        std::string hoverTint;
        std::function<void()> onClick;
    };

    void update(Config config) {
        Sakura::Button::Config button{
            .str = std::move(config.str),
            .size = {config.width, config.height},
            .chrome = true,
            .textScale = config.textScale,
            .rounding = PillRounding,
            .padding = {config.width > 0.0f ? 0.0f : 12.0f, 5.0f},
            .onClick = std::move(config.onClick),
            .font = Sakura::Text::Font::Display,
        };
        switch (config.tone) {
            case Tone::Primary:
                button.colorKey = "primary_btn";
                button.hoveredColorKey = "primary_btn_hovered";
                button.activeColorKey = "primary_btn_hovered";
                button.textColorKey = "primary_btn_text";
                break;
            case Tone::Action:
                button.colorKey = "action_btn";
                button.hoveredColorKey = "action_btn_hovered";
                button.activeColorKey = "action_btn_active";
                button.textColorKey = "action_btn_text";
                break;
            case Tone::Card:
                button.colorKey = "card_surface";
                button.hoveredColorKey = "card_surface_hovered";
                button.activeColorKey = "card_surface_hovered";
                button.textColorKey = "text_primary";
                if (!config.hoverTint.empty()) {
                    button.hoveredColorKey = jst::fmt::format("tint_{}_hover", config.hoverTint);
                    button.activeColorKey = button.hoveredColorKey;
                    button.hoveredTextColorKey = jst::fmt::format("tint_{}_text", config.hoverTint);
                }
                break;
        }
        Button::update(std::move(button));
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_CARD_BUTTON_HH
