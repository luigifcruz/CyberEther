#include "registry_plugin.hh"

namespace Jetstream {

std::string RegistryPluginView::VersionLabel(const Plugin& plugin) {
    if (plugin.version.empty()) {
        return "Unknown version";
    }
    return jst::fmt::format("Version {}", plugin.version);
}

std::string RegistryPluginView::SummaryLabel(const Plugin& plugin) {
    std::vector<std::string> notes;
    if (!plugin.status.label.empty() && plugin.status.label != "Loaded") {
        notes.push_back(plugin.status.label);
    }
    if (!plugin.update.empty()) {
        notes.push_back(plugin.update == plugin.version ? "New build available"
                                                        : jst::fmt::format("Version {} available", plugin.update));
    }
    if (notes.empty()) {
        return VersionLabel(plugin);
    }
    return jst::fmt::format("{} ({})", VersionLabel(plugin), jst::fmt::join(notes, ", "));
}

std::string RegistryPluginView::UpdateLabel(const Plugin& plugin) {
    return plugin.updating ? "Updating..." : "Update";
}

void RegistryPluginView::update(Config config) {
    this->config = std::move(config);
    const auto& plugin = this->config.plugin;
    breadcrumb.update({
        .root = "Registry",
        .current = plugin.name,
        .onRoot = this->config.onBack,
    });
    updateButton.update({
        .str = UpdateLabel(plugin),
        .icon = ICON_LC_DOWNLOAD,
        .tone = CompactButton::Tone::Primary,
        .disabled = plugin.updating,
        .onClick = [this]() {
            if (this->config.onUpdate) {
                this->config.onUpdate(this->config.plugin.id);
            }
        },
    });
    reloadButton.update({
        .str = "Reload",
        .icon = ICON_LC_REFRESH_CW,
        .onClick = [this]() {
            if (this->config.onReload) {
                this->config.onReload(this->config.plugin.id);
            }
        },
    });
    removeButton.update({
        .str = "Remove",
        .icon = ICON_LC_TRASH_2,
        .tone = CompactButton::Tone::Danger,
        .onClick = [this]() {
            if (this->config.onRemove) {
                this->config.onRemove(this->config.plugin.id);
            }
        },
    });
    std::vector<Sakura::Stack::Item> actionItems;
    if (!plugin.update.empty()) {
        actionItems.push_back({.child = &updateButton});
    }
    actionItems.push_back({.child = plugin.temporary ? static_cast<Sakura::Component*>(&reloadButton) : &removeButton});
    actions.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = std::move(actionItems),
    });
    header.update({
        .title = plugin.name,
        .description = plugin.summary.empty() ? VersionLabel(plugin) : plugin.summary,
        .action = &actions,
    });
    statusBadge.update(plugin.status);
    const std::array<std::string, 4> values = {
        plugin.version.empty() ? "Unknown" : plugin.version,
        plugin.cyberether.empty() ? "Unknown" : plugin.cyberether,
        plugin.author,
        plugin.license,
    };
    for (U64 index = 0; index < values.size(); ++index) {
        aboutValues[index].update({
            .str = values[index],
            .font = Sakura::Text::Font::Sans,
        });
    }
    std::vector<Sakura::SettingCard::Row> aboutRows = {
        {.label = "Status", .control = &statusBadge},
        {.label = "ID", .description = plugin.id},
        {.label = "Version", .control = &aboutValues[0]},
        {.label = "Built For", .control = &aboutValues[1]},
    };
    if (!plugin.author.empty()) {
        aboutRows.push_back({.label = "Author", .control = &aboutValues[2]});
    }
    if (!plugin.license.empty()) {
        aboutRows.push_back({.label = "License", .control = &aboutValues[3]});
    }
    aboutRows.push_back({
        .label = plugin.temporary ? "Source" : "Location",
        .description = plugin.path,
    });
    std::string about = plugin.description;
    if (about.empty()) {
        about = plugin.temporary
            ? "Details reported by the plugin bundle. This plugin was loaded with --plugin for this run only."
            : "Details reported by the plugin bundle.";
    }
    aboutGroup.update({
        .title = "About",
        .description = std::move(about),
        .rows = std::move(aboutRows),
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
