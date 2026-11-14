#include "registry.hh"

namespace Jetstream {

void RegistrySettingsView::update(Config config) {
    this->config = std::move(config);
    const PluginRow* plugin = selectedPlugin();
    RegistryPage page = this->config.page;
    if (page == RegistryPage::Plugin && !plugin) {
        page = RegistryPage::Overview;
    }
    if (page == RegistryPage::AddPlugin && lastPage != RegistryPage::AddPlugin) {
        addPage.reset();
    }
    lastPage = page;

    switch (page) {
        case RegistryPage::Overview:
            updateOverview();
            break;
        case RegistryPage::AddPlugin:
            addPage.update({
                .onBrowse = this->config.onBrowse,
                .onRegister = this->config.onRegister,
                .onBack = back(),
            });
            stack.update({.items = {{.child = &addPage.content()}}});
            break;
        case RegistryPage::Plugin:
            pluginPage.update({
                .plugin = *plugin,
                .onReload = this->config.onReloadPlugin,
                .onRemove = this->config.onRemovePlugin,
                .onBack = back(),
            });
            stack.update({.items = {{.child = &pluginPage.content()}}});
            break;
    }
}

Sakura::Component& RegistrySettingsView::content() {
    return stack;
}

void RegistrySettingsView::updateOverview() {
    header.update({
        .title = "Registry",
        .description = "Block domains available in this installation and the plugins that provide them.",
    });

    std::vector<Sakura::SettingCard::Row> domainRows;
    domainRows.reserve(config.domains.size());
    for (const auto& domain : config.domains) {
        domainRows.push_back({
            .label = domain.domain,
            .description = jst::fmt::format("{}", jst::fmt::join(domain.blocks, ", ")),
        });
    }
    domainGroup.update({
        .title = "Domains",
        .description = "Registered block domains and the blocks available in each domain.",
        .rows = std::move(domainRows),
        .empty = "No block domains are registered.",
    });

    while (entries.size() < config.plugins.size()) {
        entries.push_back(std::make_unique<PluginEntry>());
    }
    std::vector<Sakura::SettingCard::Row> pluginRows;
    pluginRows.reserve(config.plugins.size());
    for (U64 index = 0; index < config.plugins.size(); ++index) {
        const auto& plugin = config.plugins[index];
        auto& entry = *entries[index];
        entry.chevron.update({
            .str = ICON_LC_CHEVRON_RIGHT,
            .tone = Sakura::Text::Tone::Secondary,
            .scale = 0.8f,
        });
        pluginRows.push_back({
            .label = plugin.name,
            .description = RegistryPluginView::VersionLabel(plugin),
            .control = &entry.chevron,
            .onClick = [this, path = plugin.path]() {
                navigate(RegistryPage::Plugin, path);
            },
        });
    }
    addButton.update({
        .str = "Add",
        .icon = ICON_LC_PLUS,
        .tone = CompactButton::Tone::Primary,
        .onClick = [this]() {
            navigate(RegistryPage::AddPlugin, "");
        },
    });
    pluginGroup.update({
        .title = "Plugins",
        .description = "Load additional block domains from plugins at startup.",
        .rows = std::move(pluginRows),
        .empty = "No plugins registered yet. Use Add to load one.",
        .action = &addButton,
    });

    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &header},
            {.child = &domainGroup},
            {.child = &pluginGroup},
        },
    });
}

Sakura::Callback<void()> RegistrySettingsView::back() {
    return [this]() {
        navigate(RegistryPage::Overview, "");
    };
}

void RegistrySettingsView::navigate(RegistryPage page, const std::string& path) const {
    if (config.onNavigate) {
        config.onNavigate(page, path);
    }
}

const RegistrySettingsView::PluginRow* RegistrySettingsView::selectedPlugin() const {
    for (const auto& plugin : config.plugins) {
        if (plugin.path == config.plugin) {
            return &plugin;
        }
    }
    return nullptr;
}

}  // namespace Jetstream
