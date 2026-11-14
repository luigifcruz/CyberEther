#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH

#include "../components/compact_button.hh"
#include "../components/section_group.hh"
#include "../components/status_badge.hh"
#include "intelligence_model.hh"
#include "intelligence_provider.hh"
#include "components/settings_header.hh"

#include "../../model/ui.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceSettingsView {
    struct ProviderRow {
        std::string name;
        StatusBadge::Config status;
    };

    struct ModelRow {
        std::string provider;
        std::string id;
        std::string label;
        std::vector<std::string> tags;
    };

    struct VoiceRow {
        StatusBadge::Config status;
        bool available = false;
        std::vector<std::string> backends;
        std::optional<U64> backend;
        std::vector<std::string> models;
        std::optional<U64> model;
        std::vector<std::string> voices;
        std::optional<U64> voice;
    };

    struct McpRow {
        Settings::Intelligence::Mcp settings;
        std::string url;
        std::string error;
    };

    struct Config {
        IntelligencePage page = IntelligencePage::Overview;
        IntelligenceProviderView::Config provider;
        IntelligenceModelView::Config model;
        bool enabled = true;
        std::vector<ProviderRow> providers;
        std::vector<ModelRow> models;
        VoiceRow voice;
        McpRow mcp;
        Sakura::Callback<void(IntelligencePage)> onNavigate;
        Sakura::Callback<void(bool)> onEnabledChange;
        Sakura::Callback<void(U64)> onVoiceBackendSelect;
        Sakura::Callback<void(U64)> onVoiceModelSelect;
        Sakura::Callback<void(U64)> onVoiceSelect;
        Sakura::Callback<void()> onOpenAddProvider;
        Sakura::Callback<void(const std::string&)> onEditProvider;
        Sakura::Callback<void(const std::string&)> onRemoveProvider;
        Sakura::Callback<void()> onOpenAddModel;
        Sakura::Callback<void(const ModelRow&)> onRemoveModel;
        Sakura::Callback<void(const Settings::Intelligence::Mcp&)> onMcpChange;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    struct ProviderControls {
        StatusBadge status;
        CompactButton edit;
        CompactButton remove;
        Sakura::Stack row;
    };

    struct ModelControls {
        std::vector<std::unique_ptr<StatusBadge>> tags;
        CompactButton remove;
        Sakura::Stack row;
    };

    Sakura::Callback<void()> back();
    void updateProviders();
    void updateModels();
    void updateVoice();
    void updateMcp();
    void changeMcp(const Settings::Intelligence::Mcp& settings) const;
    static bool portValue(const std::string& label, U16& port);
    static Sakura::Combo::Config selection(std::string id,
                                           const std::vector<std::string>& options,
                                           std::optional<U64> selected,
                                           bool disabled,
                                           const Sakura::Callback<void(U64)>& onSelect);

    Config config;
    SettingsHeader header;
    Sakura::Toggle enabledToggle;
    SectionGroup assistantGroup;
    std::vector<std::unique_ptr<ProviderControls>> providerControls;
    CompactButton addProviderButton;
    SectionGroup providersGroup;
    std::vector<std::unique_ptr<ModelControls>> modelControls;
    CompactButton addModelButton;
    SectionGroup modelsGroup;
    Sakura::Combo voiceBackendCombo;
    Sakura::Combo voiceModelCombo;
    Sakura::Combo voiceNameCombo;
    StatusBadge voiceStatus;
    SectionGroup voiceGroup;
    Sakura::Toggle mcpToggle;
    StatusBadge mcpStatus;
    Sakura::PillInput mcpAddressInput;
    Sakura::PillInput mcpPortInput;
    SectionGroup mcpGroup;
    IntelligenceProviderView providerPage;
    IntelligenceModelView modelPage;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH
