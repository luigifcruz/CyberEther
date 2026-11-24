#include "registry.hh"

namespace Jetstream {

RegistrySettingsPresenter::RegistrySettingsPresenter(const PresenterContext& context) : context(context) {}

RegistrySettingsView::Config RegistrySettingsPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& registry = context.state.registry;

    std::map<std::string, std::vector<std::string>> domains;
    for (const auto& block : registry.blocks) {
        domains[block.domain].push_back(block.title.empty() ? block.type : block.title);
    }

    RegistrySettingsView::Config config{
        .page = context.state.settings.registryPage,
        .plugin = context.state.settings.registryPlugin,
        .updatePolicy = context.state.plugins.updatePolicy,
        .onNavigate = [enqueue](RegistryPage page, const std::string& plugin) {
            enqueue(MailSetRegistryPage{.page = page, .plugin = plugin});
        },
        .onBrowse = [enqueue](const std::string& currentPath, Sakura::Callback<void(std::string)> onSelect) {
            enqueue(MailBrowseConfigPath{
                .path = currentPath,
                .save = false,
                .extensions = {"cep"},
                .onSelect = std::move(onSelect),
            });
        },
        .onRegister = [enqueue](const std::string& path) {
            enqueue(MailInstallPlugin{.path = path});
        },
        .onRemovePlugin = [enqueue](const std::string& id) {
            enqueue(MailRemovePlugin{.id = id});
        },
        .onReloadPlugin = [enqueue](const std::string& id) {
            enqueue(MailReloadPlugin{.id = id});
        },
        .onUpdatePlugin = [enqueue](const std::string& id) {
            enqueue(MailUpdateGalleryPlugins{.ids = {id}});
        },
        .onUpdatePolicyChange = [enqueue](const std::string& value) {
            enqueue(MailSetPluginUpdatePolicy{.value = value});
        },
        .onBrowsePlugins = [enqueue]() {
            enqueue(MailSetWelcomeSection{.section = WelcomeSection::Plugins});
            enqueue(MailOpenStartTab{});
        },
    };

    config.domains.reserve(domains.size());
    for (auto& [domain, blocks] : domains) {
        std::sort(blocks.begin(), blocks.end());
        config.domains.push_back({.domain = domain, .blocks = std::move(blocks)});
    }

    const auto updates = context.state.pluginUpdates();
    const auto& updating = context.state.plugins.update.ids;
    config.plugins.reserve(registry.plugins.size());
    for (const auto& info : registry.plugins) {
        const auto update = std::find_if(updates.begin(), updates.end(), [&](const auto& entry) {
            return entry.id == info.id;
        });
        config.plugins.push_back({
            .id = info.id,
            .name = info.title.empty() ? info.id : info.title,
            .summary = info.summary,
            .description = info.description,
            .version = info.version,
            .cyberether = info.cyberether,
            .author = info.author,
            .license = info.license,
            .status = Status(info),
            .path = info.temporary ? info.source : info.path,
            .update = update == updates.end() || info.temporary ? "" : update->version,
            .updating = std::find(updating.begin(), updating.end(), info.id) != updating.end(),
            .temporary = info.temporary,
            .blocks = blocks(info),
            .examples = examples(info),
        });
    }

    return config;
}

StatusBadge::Config RegistrySettingsPresenter::Status(const Plugin::Info& info) {
    switch (info.status) {
        case Plugin::Status::Loaded:
            if (!info.pending.empty()) {
                return {.label = jst::fmt::format("Version {} ready on restart", info.pending),
                        .tone = StatusBadge::Tone::Success};
            }
            return {.label = info.temporary ? "Loaded for this run" : "Loaded", .tone = StatusBadge::Tone::Success};
        case Plugin::Status::Restart:
            return {.label = "Ready on restart"};
        case Plugin::Status::Incompatible:
            return {.label = "Replacement file needed", .tone = StatusBadge::Tone::Warning};
        case Plugin::Status::Waiting:
            return {.label = "Waiting for a compatible build", .tone = StatusBadge::Tone::Warning};
        case Plugin::Status::Failed:
            return {.label = "Failed to load", .tone = StatusBadge::Tone::Error};
    }
    return {.label = "Unknown"};
}

std::vector<RegistrySettingsView::PluginItem> RegistrySettingsPresenter::blocks(const Plugin::Info& info) const {
    const auto& registered = context.state.registry.blocks;
    std::vector<RegistrySettingsView::PluginItem> items;
    items.reserve(info.blocks.size());
    for (const auto& type : info.blocks) {
        const auto block = std::find_if(registered.begin(), registered.end(), [&](const auto& entry) {
            return entry.type == type;
        });
        if (block == registered.end()) {
            items.push_back({.title = type});
            continue;
        }
        items.push_back({
            .title = block->title.empty() ? type : block->title,
            .summary = block->summary,
        });
    }
    return items;
}

std::vector<RegistrySettingsView::PluginItem> RegistrySettingsPresenter::examples(const Plugin::Info& info) const {
    std::vector<RegistrySettingsView::PluginItem> items;
    items.reserve(info.examples.size());
    for (const auto& key : info.examples) {
        const auto* flowgraph = context.state.registry.flowgraph(key);
        if (flowgraph == nullptr) {
            items.push_back({.title = key});
            continue;
        }
        items.push_back({
            .title = flowgraph->title.empty() ? key : flowgraph->title,
            .summary = flowgraph->summary,
        });
    }
    return items;
}

}  // namespace Jetstream
