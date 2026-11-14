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

    explicit RegistrySettingsPresenter(const PresenterContext& context);
    RegistrySettingsView::Config build() const;

 private:
    static StatusBadge::Config Status(const DefaultCompositorState::RegistryState::PluginEntry& entry);
    std::vector<RegistrySettingsView::PluginItem> blocks(const Plugin::Info& info) const;
    std::vector<RegistrySettingsView::PluginItem> examples(const Plugin::Info& info) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REGISTRY_HH
