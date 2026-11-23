#include "missing_plugins.hh"

namespace Jetstream {

void MissingPluginsView::update(Config config) {
    this->config = std::move(config);

    using Text = Sakura::Text;

    const auto names = Jetstream::Text::Join(this->config.plugins, ", ");
    const bool single = this->config.plugins.size() == 1;

    icon.update({
        .str = ICON_LC_PACKAGE,
        .colorKey = "text_primary",
        .scale = Sakura::Typography::Scale::Heading,
    });
    title.update({
        .str = single ? "Plugin Needed" : "Plugins Needed",
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
        .str = jst::fmt::format("This flowgraph uses {} {}, which {} not installed. Install {} from the "
                                "CyberEther gallery to open it. Plugins run native code with your permissions.",
                                single ? "the plugin" : "the plugins",
                                names,
                                single ? "is" : "are",
                                single ? "it" : "them"),
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

    cancel.update({
        .label = "Cancel",
        .hoverColorKey = "chrome_button_hover",
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onCancel) {
                this->config.onCancel();
            }
        },
    });
    install.update({
        .icon = ICON_LC_DOWNLOAD,
        .label = this->config.installing ? "Installing..." : (single ? "Install Plugin" : "Install Plugins"),
        .colorKey = "action_btn_text",
        .backgroundColorKey = "action_btn",
        .hoverColorKey = "action_btn_hovered",
        .raised = true,
        .disabled = this->config.installing,
        .height = ButtonHeight,
        .onClick = [this]() {
            if (!this->config.installing && this->config.onInstall) {
                this->config.onInstall();
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
            {.child = &install},
        },
    });

    stack.update({
        .spacing = Sakura::Spacing::Large,
        .items = {
            {.child = &header},
            {.child = &footer},
        },
    });
    panel.update({
        .id = "MissingPluginsModal",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& MissingPluginsView::component() {
    return panel;
}

}  // namespace Jetstream
