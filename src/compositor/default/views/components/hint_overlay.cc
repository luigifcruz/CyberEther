#include "hint_overlay.hh"

namespace Jetstream {

void HintOverlay::update(Config config) {
    using Text = Sakura::Text;

    this->config = std::move(config);
    icon.update({
        .str = this->config.icon,
        .colorKey = "text_primary",
        .scale = Sakura::Typography::Scale::Headline,
    });
    title.update({
        .str = this->config.title,
        .font = Text::Font::DisplaySemiBold,
        .scale = Sakura::Typography::Scale::Headline,
    });
    heading.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &icon},
            {.child = &title},
        },
    });
    subtitle.update({
        .str = this->config.subtitle,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Body,
        .lineHeight = 1.05f,
    });
    updateSteps();
    updateActions();

    std::vector<Sakura::Stack::Item> items = {
        {.child = &heading},
        {.child = &subtitle},
        {.child = &headerSpacing},
    };
    if (!this->config.steps.empty()) {
        items.push_back({.child = &steps});
        items.push_back({.child = &stepsSpacing});
    }
    for (U64 i = 0; i < this->config.actions.size(); ++i) {
        items.push_back({.child = actionButtons[i].get()});
    }
    headerSpacing.update({.lines = 2});
    stepsSpacing.update({.lines = 2});
    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .centerCross = false,
        .items = std::move(items),
    });
    card.update({
        .content = &stack,
        .padding = Sakura::Spacing::XLarge,
        .rounding = Sakura::Rounding::Popup,
        .border = true,
        .chrome = true,
        .shadow = true,
        .colorKey = "card_surface",
        .hoveredColorKey = "card_surface",
    });
}

Sakura::Component& HintOverlay::component() {
    return card;
}

Extent2D<F32> HintOverlay::size() const {
    return {config.width, 0.0f};
}

void HintOverlay::updateSteps() {
    using Text = Sakura::Text;

    while (stepRows.size() < config.steps.size()) {
        stepRows.push_back(std::make_unique<Step>());
    }
    std::vector<Sakura::Stack::Item> rows;
    for (U64 i = 0; i < config.steps.size(); ++i) {
        auto& step = *stepRows[i];
        step.number.update({
            .str = jst::fmt::format("{}", i + 1),
            .font = Text::Font::DisplaySemiBold,
            .align = Text::Align::Center,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Small,
        });
        step.badge.update({
            .content = &step.number,
            .size = {BadgeSize, BadgeSize},
            .padding = (BadgeSize - BadgeText) * 0.5f,
            .rounding = BadgeSize * 0.5f,
            .border = false,
            .colorKey = "pill_bg",
            .hoveredColorKey = "pill_bg",
        });
        step.text.update({
            .str = config.steps[i],
            .font = Text::Font::Sans,
            .wrapped = true,
            .scale = Sakura::Typography::Scale::Lead,
            .weight = 0.15f,
        });
        step.row.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Medium,
            .items = {
                {.child = &step.badge},
                {.child = &step.text, .grow = 1.0f, .shrink = 1.0f},
            },
        });
        rows.push_back({.child = &step.row});
    }
    steps.update({
        .spacing = Sakura::Spacing::Medium,
        .centerCross = false,
        .items = std::move(rows),
    });
}

void HintOverlay::updateActions() {
    while (actionButtons.size() < config.actions.size()) {
        actionButtons.push_back(std::make_unique<CardButton>());
    }
    for (U64 i = 0; i < config.actions.size(); ++i) {
        const auto& action = config.actions[i];
        actionButtons[i]->update({
            .str = action.label,
            .tone = action.tone,
            .height = ButtonHeight,
            .textScale = Sakura::Typography::Scale::Title,
            .onClick = action.onClick,
        });
    }
}

}  // namespace Jetstream
