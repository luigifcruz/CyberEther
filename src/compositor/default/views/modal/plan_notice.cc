#include "plan_notice.hh"

namespace Jetstream {

void PlanNoticeView::update(Config config) {
    this->config = std::move(config);

    using Text = Sakura::Text;

    topSpacing.update({.lines = 1});
    middleSpacing.update({.lines = 1});
    icon.update({
        .str = ICON_LC_SPARKLES,
        .align = Text::Align::Center,
        .colorKey = "text_primary",
        .scale = IconScale,
    });
    title.update({
        .str = "You're using your ChatGPT plan!",
        .font = Text::Font::Display,
        .align = Text::Align::Center,
        .scale = Sakura::Typography::Scale::Headline,
    });
    description.update({
        .str = "Eligible usage in this app uses your ChatGPT plan. Manage usage in your ChatGPT settings. "
               "Next, add models from your plan.",
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .align = Text::Align::Center,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Lead,
    });
    header.update({
        .spacing = Sakura::Spacing::Medium,
        .items = {
            {.child = &icon},
            {.child = &title},
            {.child = &description},
        },
    });

    manage.update({
        .str = ICON_LC_EXTERNAL_LINK "  Manage Usage",
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onManageUsage) {
                this->config.onManageUsage();
            }
        },
    });
    addModels.update({
        .str = ICON_LC_PLUS "  Add Models",
        .tone = CardButton::Tone::Action,
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onAddModels) {
                this->config.onAddModels();
            }
        },
    });
    footer.update({
        .columns = 2,
        .cellPadding = {3.0f, 0.0f},
        .outerPadding = false,
        .items = {&manage, &addModels},
    });

    stack.update({
        .spacing = Sakura::Spacing::Large,
        .items = {
            {.child = &topSpacing},
            {.child = &header},
            {.child = &middleSpacing},
            {.child = &footer},
        },
    });
    panel.update({
        .id = "PlanNotice",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& PlanNoticeView::component() {
    return panel;
}

}  // namespace Jetstream
