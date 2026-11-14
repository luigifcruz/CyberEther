#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_GENERAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_GENERAL_HH

#include "../components/option_stepper.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/memory/types.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <cmath>
#include <string>
#include <vector>

namespace Jetstream {

struct GeneralSettingsView {
    struct Config {
        std::vector<std::string> themes;
        std::string currentThemeKey;
        F32 interfaceScale = 1.0f;
        std::string renderer;
        bool backgroundParticles = false;
        bool showTips = true;
        Sakura::Callback<void(const std::string&)> onThemeChange;
        Sakura::Callback<void(F32)> onInterfaceScaleChange;
        Sakura::Callback<void(DeviceType)> onRendererChange;
        Sakura::Callback<void(bool)> onBackgroundParticlesChange;
        Sakura::Callback<void(bool)> onShowTipsChange;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    static DeviceType rendererValue(const std::string& label);
    static std::vector<std::string> rendererLabels();
    static const std::vector<F32>& scaleValues();
    static std::vector<std::string> scaleLabels();
    static U64 scaleIndex(F32 scale);

    Config config;
    SettingsHeader header;
    Sakura::PillTabs themeTabs;
    OptionStepper scaleStepper;
    Sakura::PillTabs rendererTabs;
    Sakura::Toggle particlesToggle;
    Sakura::Toggle tipsToggle;
    SectionGroup appearanceGroup;
    SectionGroup renderingGroup;
    Sakura::Stack stack;
    std::string renderer;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_GENERAL_HH
