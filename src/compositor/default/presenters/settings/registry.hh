#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REGISTRY_HH

#include "../context.hh"
#include "plugin_labels.hh"

#include "../../model/messages.hh"
#include "../../views/settings/registry.hh"

#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace Jetstream {

struct RegistrySettingsPresenter {
    const PresenterContext& context;

    explicit RegistrySettingsPresenter(const PresenterContext& context) : context(context) {}

    RegistrySettingsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& registry = context.state.registry;

        std::map<std::string, std::vector<std::string>> domains;
        for (const auto& block : registry.blocks) {
            domains[block.domain].push_back(block.title.empty() ? block.type : block.title);
        }

        RegistrySettingsView::Config config{
            .page = context.state.settings.registryPage,
            .plugin = context.state.settings.registryPlugin,
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
                enqueue(MailAddPluginPath{.path = path});
            },
            .onRemovePlugin = [enqueue](const std::string& path) {
                enqueue(MailRemovePluginPath{.path = path});
            },
            .onReloadPlugin = [enqueue](const std::string& path) {
                enqueue(MailReloadPlugin{.path = path});
            },
        };

        config.domains.reserve(domains.size());
        for (auto& [domain, blocks] : domains) {
            std::sort(blocks.begin(), blocks.end());
            config.domains.push_back({.domain = domain, .blocks = std::move(blocks)});
        }

        config.plugins.reserve(registry.registered.size());
        for (const auto& entry : registry.registered) {
            if (entry.info.has_value()) {
                const auto& info = *entry.info;
                config.plugins.push_back({
                    .name = PluginLabels::Name(entry.path, info.name),
                    .version = info.version,
                    .status = {
                        .label = info.status.empty() ? "Loaded" : info.status,
                        .tone = StatusBadge::Tone::Success,
                    },
                    .path = entry.path,
                    .minimumVersion = info.minimumJetstreamVersion,
                    .blocks = blocks(info),
                    .examples = examples(info),
                });
                continue;
            }
            config.plugins.push_back({
                .name = PluginLabels::Name(entry.path),
                .status = Status(entry),
                .path = entry.path,
            });
        }

        return config;
    }

 private:
    static StatusBadge::Config Status(const DefaultCompositorState::RegistryState::PluginEntry& entry) {
        if (Platform::PathFromUtf8(entry.path).extension() != ".cep") {
            return {.label = "Invalid extension", .tone = StatusBadge::Tone::Warning};
        }
        if (!entry.exists) {
            return {.label = "Missing", .tone = StatusBadge::Tone::Warning};
        }
        return {.label = "Not loaded"};
    }

    std::vector<RegistrySettingsView::PluginItem> blocks(const Plugin::Info& info) const {
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

    std::vector<RegistrySettingsView::PluginItem> examples(const Plugin::Info& info) const {
        const auto& registered = context.state.registry.flowgraphs;
        std::vector<RegistrySettingsView::PluginItem> items;
        items.reserve(info.examples.size());
        for (const auto& key : info.examples) {
            const auto flowgraph = std::find_if(registered.begin(), registered.end(), [&](const auto& entry) {
                return entry.key == key;
            });
            if (flowgraph == registered.end()) {
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
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REGISTRY_HH
