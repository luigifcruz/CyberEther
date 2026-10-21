#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REGISTRY_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/registry.hh"

#include "jetstream/registry.hh"
#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Jetstream {

struct RegistrySettingsPresenter {
    const PresenterContext& context;

    explicit RegistrySettingsPresenter(const PresenterContext& context) : context(context) {}

    static std::string normalizePluginPath(const std::string& path) {
        std::error_code ec;
        const auto inputPath = Platform::PathFromUtf8(path);
        const auto canonical = std::filesystem::weakly_canonical(inputPath, ec);
        if (!ec) {
            return Platform::PathToUtf8(canonical);
        }

        ec.clear();
        const auto absolute = std::filesystem::absolute(inputPath, ec);
        if (!ec) {
            return Platform::PathToUtf8(absolute.lexically_normal());
        }

        return path;
    }

    static std::string fileName(const std::string& path) {
        const auto name = Platform::PathToUtf8(Platform::PathFromUtf8(path).filename());
        return name.empty() ? path : name;
    }

    static std::vector<RegistrySettingsPanel::PluginItem> pluginBlocks(const Plugin::Info& info) {
        std::vector<RegistrySettingsPanel::PluginItem> items;
        items.reserve(info.blocks.size());
        for (const auto& type : info.blocks) {
            const auto blocks = Registry::ListAvailableBlocks(type);
            if (blocks.empty()) {
                items.push_back({.title = type});
                continue;
            }
            const auto& block = blocks.front();
            items.push_back({
                .title = block.title.empty() ? type : block.title,
                .summary = block.summary,
            });
        }
        return items;
    }

    static std::vector<RegistrySettingsPanel::PluginItem> pluginExamples(const Plugin::Info& info) {
        std::vector<RegistrySettingsPanel::PluginItem> items;
        items.reserve(info.examples.size());
        for (const auto& key : info.examples) {
            const auto flowgraphs = Registry::ListAvailableFlowgraphs(key);
            if (flowgraphs.empty()) {
                items.push_back({.title = key});
                continue;
            }
            const auto& flowgraph = flowgraphs.front();
            items.push_back({
                .title = flowgraph.title.empty() ? key : flowgraph.title,
                .summary = flowgraph.summary,
            });
        }
        return items;
    }

    RegistrySettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        std::map<std::string, std::vector<std::string>> domains;

        for (const auto& block : Registry::ListAvailableBlocks("")) {
            domains[block.domain].push_back(block.title.empty() ? block.type : block.title);
        }

        std::vector<RegistrySettingsPanel::DomainRow> rows;
        rows.reserve(domains.size());
        for (auto& entry : domains) {
            std::sort(entry.second.begin(), entry.second.end());
            rows.push_back({
                .domain = entry.first,
                .blocks = entry.second,
            });
        }

        std::vector<RegistrySettingsPanel::PluginRow> plugins;
        Settings settings;
        if (Settings::Get(settings) == Result::SUCCESS) {
            std::map<std::string, Plugin::Info> loadedPlugins;
            for (const auto& plugin : Plugin::List()) {
                loadedPlugins[normalizePluginPath(plugin.path)] = plugin;
            }

            plugins.reserve(settings.registry.plugins.size());
            for (const auto& path : settings.registry.plugins) {
                const auto normalizedPath = normalizePluginPath(path);
                const auto loaded = loadedPlugins.find(normalizedPath);

                if (loaded != loadedPlugins.end()) {
                    const auto& info = loaded->second;
                    plugins.push_back({
                        .name = info.name.empty() ? fileName(path) : info.name,
                        .version = info.version.empty() ? "-" : info.version,
                        .status = info.status.empty() ? "Loaded" : info.status,
                        .path = path,
                        .minimumVersion = info.minimumJetstreamVersion,
                        .blocks = pluginBlocks(info),
                        .examples = pluginExamples(info),
                    });
                    continue;
                }

                std::string status = "Not loaded";
                if (Platform::PathFromUtf8(path).extension() != ".cep") {
                    status = "Invalid extension";
                } else if (!std::filesystem::exists(Platform::PathFromUtf8(path))) {
                    status = "Missing";
                }

                plugins.push_back({
                    .name = fileName(path),
                    .version = "-",
                    .status = status,
                    .path = path,
                });
            }
        }

        return RegistrySettingsPanel::Config{
            .domains = rows,
            .plugins = plugins,
            .page = context.state.settings.registryPage,
            .plugin = context.state.settings.registryPlugin,
            .onNavigate = [enqueue](RegistryPage page, const std::string& plugin) {
                enqueue(MailSetRegistryPage{.page = page, .plugin = plugin});
            },
            .onBrowse = [enqueue](const std::string& currentPath, std::function<void(std::string)> onSelect) {
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
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REGISTRY_HH
