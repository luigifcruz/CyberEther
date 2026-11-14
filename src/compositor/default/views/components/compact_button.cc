#include "compact_button.hh"

namespace Jetstream {

void CompactButton::update(Config config) {
    Sakura::PillButton::Config pill{
        .icon = std::move(config.icon),
        .label = std::move(config.str),
        .backgroundColorKey = "pill_field_bg",
        .hoverColorKey = "pill_hover_bg",
        .outlineColorKey = "tile_border",
        .raised = true,
        .disabled = config.disabled,
        .height = 34.0f,
        .textSize = 15.0f,
        .padLeft = 12.0f,
        .padRight = 15.0f,
        .onClick = std::move(config.onClick),
    };
    switch (config.tone) {
        case Tone::Default:
            break;
        case Tone::Primary:
            pill.colorKey = "action_btn_text";
            pill.backgroundColorKey = "action_btn";
            pill.hoverColorKey = "action_btn_hovered";
            pill.outlineColorKey = "";
            break;
        case Tone::Warning:
            pill.colorKey = "warning_btn_text";
            pill.backgroundColorKey = "warning_btn";
            pill.hoverColorKey = "warning_btn_hovered";
            pill.outlineColorKey = "warning_btn_outline";
            break;
        case Tone::Danger:
            pill.colorKey = "tint_red_text";
            pill.backgroundColorKey = "tint_red_bg";
            pill.hoverColorKey = "tint_red_hover";
            pill.outlineColorKey = "tint_red_border";
            break;
    }
    PillButton::update(std::move(pill));
}

}  // namespace Jetstream
