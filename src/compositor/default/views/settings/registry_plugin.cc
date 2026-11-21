#include "registry_plugin.hh"

namespace Jetstream {

std::string RegistryPluginView::VersionLabel(const Plugin& plugin) {
    if (plugin.version.empty()) {
        return "Unknown version";
    }
    return jst::fmt::format("Version {}", plugin.version);
}

void RegistryPluginView::update(Config config) {
    this->config = std::move(config);
    const auto& plugin = this->config.plugin;
    breadcrumb.update({
        .root = "Registry",
        .current = plugin.name,
        .onRoot = this->config.onBack,
    });
    reloadButton.update({
        .str = "Reload",
        .icon = ICON_LC_REFRESH_CW,
        .onClick = [this]() {
            if (this->config.onReload) {
                this->config.onReload(this->config.plugin.path);
            }
        },
    });
    removeButton.update({
        .str = "Remove",
        .icon = ICON_LC_TRASH_2,
        .tone = CompactButton::Tone::Danger,
        .onClick = [this]() {
            if (this->config.onRemove) {
                this->config.onRemove(this->config.plugin.path);
            }
        },
    });
    actions.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &reloadButton},
            {.child = &removeButton},
        },
    });
    header.update({
        .title = plugin.name,
        .description = VersionLabel(plugin),
        .action = &actions,
    });
    statusBadge.update(plugin.status);
    const std::array<std::string, 2> values = {
        plugin.version.empty() ? "Unknown" : plugin.version,
        plugin.minimumVersion.empty() ? "Unknown" : plugin.minimumVersion,
    };
    for (U64 index = 0; index < values.size(); ++index) {
        aboutValues[index].update({
            .str = values[index],
            .font = Sakura::Text::Font::Sans,
        });
    }
    aboutGroup.update({
        .title = "About",
        .description = "Details reported by the plugin bundle.",
        .rows = {
            {.label = "Status", .control = &statusBadge},
            {.label = "Version", .control = &aboutValues[0]},
            {.label = "Minimum CyberEther", .control = &aboutValues[1]},
            {.label = "Location", .description = plugin.path},
        },
    });
    blocksGroup.update({
        .title = "Blocks",
        .description = "Blocks this plugin adds to the block catalog.",
        .rows = Rows(plugin.blocks),
        .empty = "This plugin does not register any blocks.",
    });
    examplesGroup.update({
        .title = "Examples",
        .description = "Example flowgraphs bundled with this plugin.",
        .rows = Rows(plugin.examples),
        .empty = "This plugin does not include any examples.",
    });
    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &breadcrumb},
            {.child = &header},
            {.child = &aboutGroup},
            {.child = &blocksGroup},
            {.child = &examplesGroup},
        },
    });
}

Sakura::Component& RegistryPluginView::content() {
    return stack;
}

std::vector<Sakura::SettingCard::Row> RegistryPluginView::Rows(const std::vector<Item>& items) {
    std::vector<Sakura::SettingCard::Row> rows;
    rows.reserve(items.size());
    for (const auto& item : items) {
        rows.push_back({.label = item.title, .description = item.summary});
    }
    return rows;
}

}  // namespace Jetstream
