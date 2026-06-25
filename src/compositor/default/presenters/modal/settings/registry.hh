#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REGISTRY_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/registry.hh"

#include "jetstream/registry.hh"
#include "jetstream/plugin.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace Jetstream {

struct RegistrySettingsPresenter {
    const PresenterContext& context;

    explicit RegistrySettingsPresenter(const PresenterContext& context) : context(context) {}

    static std::string normalizePluginPath(const std::string& path) {
        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(path, ec);
        if (!ec) {
            return canonical.string();
        }

        ec.clear();
        const auto absolute = std::filesystem::absolute(path, ec);
        if (!ec) {
            return absolute.lexically_normal().string();
        }

        return path;
    }

    static std::string fileName(const std::string& path) {
        const auto name = std::filesystem::path(path).filename().string();
        return name.empty() ? path : name;
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
                    });
                    continue;
                }

                std::string status = "Not loaded";
                if (std::filesystem::path(path).extension() != ".cep") {
                    status = "Invalid extension";
                } else if (!std::filesystem::exists(path)) {
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
            .onAddPlugin = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Plugin});
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

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REGISTRY_HH
