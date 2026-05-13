#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REGISTRY_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/registry.hh"

#include "jetstream/registry.hh"
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

        std::vector<RegistrySettingsPanel::LibraryRow> libraries;
        Settings settings;
        if (Settings::Get(settings) == Result::SUCCESS) {
            libraries.reserve(settings.registry.dynamicLibraries.size());
            for (const auto& path : settings.registry.dynamicLibraries) {
                libraries.push_back({
                    .path = path,
                });
            }
        }

        return RegistrySettingsPanel::Config{
            .domains = rows,
            .dynamicLibraries = libraries,
            .onAddLibrary = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Library});
            },
            .onRemoveLibrary = [enqueue](const std::string& path) {
                enqueue(MailRemoveRegistryLibraryPath{.path = path});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REGISTRY_HH
