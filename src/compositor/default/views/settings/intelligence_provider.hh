#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_PROVIDER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_PROVIDER_HH

#include "../components/compact_button.hh"
#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceProviderView {
    static constexpr const char* ChatCompletionsLabel = "Chat Completions";
    static constexpr const char* ResponsesLabel = "Responses";
    static constexpr const char* MessagesLabel = "Messages";
    static constexpr const char* KeychainLabel = "Keychain";
    static constexpr const char* EnvironmentLabel = "Environment";

    struct Preset {
        std::string label;
        std::string description;
        Settings::Intelligence::Provider provider;
    };

    struct Config {
        std::string defaultPreset;
        std::string chatPathHint;
        std::string modelsPathHint;
        U64 session = 0;
        bool editMode = false;
        Settings::Intelligence::Provider editProvider;
        std::string editPreset;
        std::vector<Preset> presets;
        bool saving = false;
        bool testRunning = false;
        bool testCompleted = false;
        bool testSuccess = false;
        U64 testRevision = 0;
        std::string testMessage;
        bool signInRunning = false;
        bool signedIn = false;
        std::string signInEmail;
        std::string signInMessage;
        std::vector<std::string> existingNames;
        Sakura::Callback<bool(const Settings::Intelligence::Provider&)> isValid;
        Sakura::Callback<void(const Settings::Intelligence::Provider&, const std::string&, U64)> onTest;
        Sakura::Callback<void(const Settings::Intelligence::Provider&, const std::string&)> onAdd;
        Sakura::Callback<void(const Settings::Intelligence::Provider&, U64)> onSignIn;
        Sakura::Callback<void()> onCancelSignIn;
        Sakura::Callback<void()> onManageUsage;
        Sakura::Callback<void()> onBack;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    void applyPreset(const std::string& label);
    void applyProvider(const Settings::Intelligence::Provider& provider);
    void markDirty();
    void reset();
    Settings::Intelligence::Provider buildProvider() const;
    void applyKeySource(const Settings::Intelligence::Provider& provider);
    U64 apiIndex() const;
    static std::string defaultChatPath(const std::string& name);
    static std::string apiName(U64 index);
    std::string keychainKey() const;
    bool keySourceValid() const;
    std::string keyDescription() const;
    std::string keyHint() const;
    bool planBilled() const;
    bool accountReady() const;
    bool accountStored() const;
    std::string accountDescription() const;
    bool nameTaken() const;
    bool testCurrent() const;
    bool testPassed() const;
    bool providerValid() const;
    bool testEnabled() const;
    bool saveEnabled() const;
    std::string saveLabel() const;
    std::string typeDescription() const;
    std::string testDescription() const;
    std::string statusBadgeLabel() const;
    StatusBadge::Config statusBadge() const;

    enum class KeySource {
        Keychain,
        Environment,
    };

    Config config;
    std::optional<U64> lastSession;
    U64 draftRevision = 0;
    bool requestReasoning = false;
    std::string typeLabel;
    std::string name;
    std::string host;
    std::string api;
    std::string chatPath;
    std::string modelsPath;
    std::string apiKey;
    std::string apiKeyEnv;
    std::string auth;
    KeySource keySource = KeySource::Keychain;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    Sakura::Combo typeCombo;
    Sakura::PillInput nameInput;
    Sakura::PillInput hostInput;
    SectionGroup endpointGroup;
    Sakura::PillTabs apiTabs;
    Sakura::PillInput chatPathInput;
    Sakura::PillInput modelsPathInput;
    Sakura::PillTabs keySourceTabs;
    Sakura::PillInput keyInput;
    Sakura::PillInput keySourceInput;
    SectionGroup apiGroup;
    CompactButton signInButton;
    CompactButton usageButton;
    SectionGroup accountGroup;
    StatusBadge testStatus;
    CompactButton testButton;
    Sakura::Stack testControls;
    Sakura::Button saveButton;
    SectionGroup connectionGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_PROVIDER_HH
