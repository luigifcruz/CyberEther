#include "placeholder.hh"

namespace Jetstream {

void AgentChatPlaceholder::update(Config config) {
    using Text = Sakura::Text;

    title.update({
        .str = config.title,
        .font = Text::Font::DisplaySemiBold,
        .wrapped = true,
        .scale = TitleScale,
        .lineHeight = 1.1f,
    });
    subtitle.update({
        .str = config.subtitle,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Caption,
        .lineHeight = 1.15f,
    });
    header.update({
        .spacing = RowSpacing,
        .centerCross = false,
        .items = {{.child = &title}, {.child = &subtitle}},
    });

    while (buttons.size() < config.suggestions.size()) {
        buttons.push_back(std::make_unique<Sakura::Button>());
    }
    std::vector<Sakura::Stack::Item> rows;
    for (U64 i = 0; i < config.suggestions.size(); ++i) {
        auto& suggestion = config.suggestions[i];
        buttons[i]->update({
            .str = jst::fmt::format("{}  {}", suggestion.icon, suggestion.label),
            .size = {Sakura::Fill, SuggestionHeight},
            .chrome = true,
            .colorKey = "agent_composer",
            .hoveredColorKey = "card_surface_hovered",
            .activeColorKey = "card_surface_hovered",
            .textColorKey = "text_secondary",
            .hoveredTextColorKey = "text_primary",
            .rounding = Sakura::Rounding::Card,
            .padding = {14.0f, 0.0f},
            .textAlign = 0.0f,
            .onClick = std::move(suggestion.onClick),
            .font = Text::Font::Sans,
        });
        rows.push_back({.child = buttons[i].get()});
    }
    if (config.onSetup) {
        setup.update({
            .str = config.setupLabel,
            .tone = CardButton::Tone::Primary,
            .height = SetupHeight,
            .textScale = 1.1f,
            .onClick = std::move(config.onSetup),
        });
        rows.push_back({.child = &setup});
    }
    actions.update({
        .spacing = RowSpacing,
        .centerCross = false,
        .items = std::move(rows),
    });
    stack.update({
        .spacing = SectionSpacing,
        .centerCross = false,
        .items = {{.child = &header}, {.child = &actions}},
    });
}

Sakura::Component& AgentChatPlaceholder::component() {
    return stack;
}

}  // namespace Jetstream
