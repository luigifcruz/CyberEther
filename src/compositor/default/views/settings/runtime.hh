#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_RUNTIME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_RUNTIME_HH

#include "../components/compact_button.hh"
#include "../components/file_field.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/runtime_context_python.hh"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RuntimeSettingsView {
    struct Config {
        std::string pythonPath;
        std::string dependencyPolicy = "prompt";
        std::vector<PythonRuntimeContext::Candidate> pythonCandidates;
        PythonRuntimeContext::Validation pythonValidation;
        bool restartRequired = false;
        Sakura::Callback<void(const std::string&)> onPythonPathChange;
        Sakura::Callback<void(const std::string&)> onDependencyPolicyChange;
        Sakura::Callback<void()> onOpenPythonEnvironmentCache;
        Sakura::Callback<void(const std::string&, Sakura::Callback<void(std::string)>)> onBrowsePythonPath;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    static const std::string& AutoRuntimeLabel();
    static const std::string& CustomPathLabel();
    static std::string dependencyPolicyLabel(const std::string& value);
    static std::string dependencyPolicyValue(const std::string& label);
    std::vector<std::string> runtimeOptions() const;
    std::string selectedRuntimeOption() const;
    void handleRuntimeOptionChange(const std::string& value);
    std::string validationBadge() const;
    std::string validationDescription() const;

    Config config;
    SettingsHeader header;
    Sakura::Combo runtimeCombo;
    FileField pythonPathField;
    StatusBadge statusBadge;
    SectionGroup pythonGroup;
    Sakura::Combo dependencyPolicyCombo;
    SectionGroup dependencyGroup;
    CompactButton cacheButton;
    SectionGroup cacheGroup;
    Sakura::Stack stack;
    bool customPathSelected = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_RUNTIME_HH
