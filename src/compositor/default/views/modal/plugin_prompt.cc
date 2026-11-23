#include "plugin_prompt.hh"

namespace Jetstream {

void PluginPromptView::update(Config config) {
    this->config = std::move(config);

    using Text = Sakura::Text;

    icon.update({
        .str = ICON_LC_PACKAGE,
        .colorKey = "text_primary",
        .scale = Sakura::Typography::Scale::Heading,
    });
    title.update({
        .str = this->config.title,
        .font = Text::Font::Display,
        .scale = Sakura::Typography::Scale::Headline,
    });
    titleRow.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &icon},
            {.child = &title},
        },
    });
    description.update({
        .str = this->config.description,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Lead,
    });
    header.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &titleRow},
            {.child = &description},
        },
    });

    table.update({
        .columns = this->config.columns,
        .rows = this->config.rows,
        .maxHeight = 240.0f,
        .wrapped = true,
    });

    cancel.update({
        .label = this->config.cancel,
        .hoverColorKey = "chrome_button_hover",
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onCancel) {
                this->config.onCancel();
            }
        },
    });
    action.update({
        .icon = ICON_LC_DOWNLOAD,
        .label = this->config.action,
        .colorKey = "action_btn_text",
        .backgroundColorKey = "action_btn",
        .hoverColorKey = "action_btn_hovered",
        .raised = true,
        .disabled = this->config.busy,
        .height = ButtonHeight,
        .onClick = [this]() {
            if (!this->config.busy && this->config.onAction) {
                this->config.onAction();
            }
        },
    });
    footerSpacer.update({});
    footer.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &footerSpacer, .grow = 1.0f},
            {.child = &cancel},
            {.child = &action},
        },
    });

    std::vector<Sakura::Stack::Item> items = {{.child = &header}};
    if (!this->config.rows.empty()) {
        items.push_back({.child = &table});
    }
    items.push_back({.child = &footer});
    stack.update({
        .spacing = Sakura::Spacing::Large,
        .items = std::move(items),
    });
    panel.update({
        .id = "PluginPromptModal",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& PluginPromptView::component() {
    return panel;
}

}  // namespace Jetstream
