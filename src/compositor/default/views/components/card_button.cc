#include "card_button.hh"

namespace Jetstream {

void CardButton::update(Config config) {
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
                if (config.tintText) {
                    button.textColorKey = button.hoveredTextColorKey;
                }
            }
            break;
    }
    Button::update(std::move(button));
}

}  // namespace Jetstream
