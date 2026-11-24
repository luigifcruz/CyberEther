#include "plugin_details.hh"

namespace Jetstream {

void PluginDetailsView::update(Config config) {
    this->config = std::move(config);

    updateHeader();
    updateActions();
    updateInfo();

    description.update({
        .id = "PluginDetailsDescription",
        .value = this->config.description,
        .padding = Sakura::Padding{DescriptionPadding, DescriptionPadding, DescriptionPadding, DescriptionPadding},
        .onLink = [this](const std::string& url) {
            if (this->config.onOpenUrl) {
                this->config.onOpenUrl(url);
            }
            return true;
        },
        .softBreaks = true,
    });
    descriptionCard.update({
        .content = &description,
        .padding = AreaInset,
        .rounding = Sakura::Rounding::Card,
        .border = true,
        .borderWidth = AreaBorder,
        .chrome = true,
        .shadow = true,
        .shadowSoftness = AreaShadowSoftness,
        .shadowDrop = AreaShadowDrop,
        .colorKey = AreaFill,
        .hoveredColorKey = AreaFill,
    });

    summaryGap.update({});
    std::vector<Sakura::Stack::Item> items = {{.child = &summaryGap, .size = SummaryHeight}};
    if (!this->config.description.empty()) {
        items.push_back({.child = &descriptionCard});
    }
    items.push_back({.child = &info});
    body.update({
        .spacing = Sakura::Spacing::Medium,
        .items = std::move(items),
    });
    scroll.update({
        .id = "plugin-details-" + this->config.id,
        .content = &body,
        .height = Sakura::Fill,
        .border = false,
        .scrollbar = false,
    });
    summaryStack.update({
        .spacing = Sakura::Spacing::Medium,
        .items = {
            {.child = &header},
            {.child = &actions},
        },
    });
    layers.update({
        .items = {
            {.child = &summaryStack, .anchor = Sakura::Anchor::TopLeft, .size = {Sakura::Fill, 0.0f}},
            {.child = &scroll},
        },
    });
    dock.update({
        .accepts = "plugin",
        .panes = {{
            .id = "plugin",
            .title = "Plugin Details",
            .group = "plugin",
            .icon = ICON_LC_PUZZLE,
            .content = &layers,
            .maximizable = false,
            .padding = PanePadding,
            .onClose = this->config.onClose,
        }},
        .resizable = false,
    });
}

Sakura::Component& PluginDetailsView::component() {
    return dock;
}

void PluginDetailsView::updateHeader() {
    using Text = Sakura::Text;

    appIcon.update({
        .str = ICON_LC_PUZZLE,
        .align = Text::Align::Center,
        .colorKey = "cyber_blue",
        .scale = IconScale,
    });
    appTile.update({
        .content = &appIcon,
        .size = {IconSize, IconSize},
        .padding = IconInset,
        .rounding = IconRounding,
        .colorKey = "card_surface",
        .borderColorKey = "setting_card_border",
    });
    title.update({
        .str = config.title,
        .font = Text::Font::DisplaySemiBold,
        .ellipsis = true,
        .scale = TitleScale,
    });
    summary.update({
        .str = config.summary,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .ellipsis = true,
        .scale = SummaryScale,
    });
    std::vector<Sakura::Stack::Item> lines = {{.child = &title}};
    if (!config.summary.empty()) {
        lines.push_back({.child = &summary});
    }
    titles.update({
        .spacing = Sakura::Spacing::XSmall,
        .centerCross = false,
        .items = std::move(lines),
    });
    header.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Medium,
        .items = {
            {.child = &appTile},
            {.child = &titles, .grow = 1.0f, .shrink = 1.0f},
        },
    });
}

void PluginDetailsView::updateActions() {
    install.update({
        .icon = ICON_LC_DOWNLOAD,
        .label = config.installLabel,
        .colorKey = "action_btn_text",
        .backgroundColorKey = "action_btn",
        .hoverColorKey = "action_btn_hovered",
        .raised = true,
        .disabled = config.installDisabled,
        .height = InstallHeight,
        .textSize = InstallText,
        .onClick = [this]() {
            if (!config.installDisabled && config.onInstall) {
                config.onInstall();
            }
        },
    });
    guide.update({
        .icon = ICON_LC_BOOK_OPEN,
        .tooltip = "How to Install",
        .backgroundColorKey = AreaFill,
        .hoverColorKey = "card_surface_hovered",
        .outlineColorKey = "setting_card_border",
        .height = InstallHeight,
        .textSize = InstallText,
        .padLeft = IconButtonPad,
        .padRight = IconButtonPad,
        .onClick = [this]() {
            if (config.onHowToInstall) {
                config.onHowToInstall();
            }
        },
    });
    copyLink.update({
        .icon = ICON_LC_LINK_2,
        .tooltip = "Copy Link",
        .backgroundColorKey = AreaFill,
        .hoverColorKey = "card_surface_hovered",
        .outlineColorKey = "setting_card_border",
        .height = InstallHeight,
        .textSize = InstallText,
        .padLeft = IconButtonPad,
        .padRight = IconButtonPad,
        .onClick = [this]() {
            if (config.onCopyLink) {
                config.onCopyLink();
            }
        },
    });
    actions.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &install, .grow = 1.0f, .shrink = 1.0f},
            {.child = &guide},
            {.child = &copyLink},
        },
    });
}

void PluginDetailsView::updateInfo() {
    using Text = Sakura::Text;

    infoTitle.update({
        .str = "Information",
        .font = Text::Font::DisplaySemiBold,
        .scale = InfoTitleScale,
    });
    while (rows.size() < config.details.size()) {
        rows.push_back(std::make_unique<Row>());
    }
    std::vector<Sakura::Stack::Item> items;
    for (U64 index = 0; index < config.details.size(); ++index) {
        const auto& detail = config.details[index];
        auto& row = *rows[index];
        row.label.update({
            .str = detail.label,
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
        });
        row.value.update({
            .str = detail.value,
            .font = Text::Font::Sans,
            .align = Text::Align::Right,
            .ellipsis = true,
        });
        row.stack.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Medium,
            .items = {
                {.child = &row.label},
                {.child = &row.value, .grow = 1.0f, .shrink = 1.0f},
            },
        });
        row.rule.update({.spacing = 0.0f});
        if (index > 0) {
            items.push_back({.child = &row.rule});
        }
        items.push_back({.child = &row.stack});
    }
    infoRows.update({
        .spacing = Sakura::Spacing::Small,
        .centerCross = false,
        .items = std::move(items),
    });
    info.update({
        .spacing = InfoTitleGap,
        .centerCross = false,
        .items = {
            {.child = &infoTitle},
            {.child = &infoRows},
        },
    });
}

}  // namespace Jetstream
