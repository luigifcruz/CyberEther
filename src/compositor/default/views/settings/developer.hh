#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_DEVELOPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_DEVELOPER_HH

#include "../components/compact_button.hh"
#include "../components/option_stepper.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include "jetstream/logger.hh"

#include <algorithm>
#include <string>

namespace Jetstream {

struct DeveloperSettingsView {
    struct Config {
        bool timingEnabled = false;
        int logLevel = JST_LOG_DEBUG_DEFAULT_LEVEL;
        Sakura::Callback<void(bool)> onTimingEnabledChange;
        Sakura::Callback<void(int)> onLogLevelChange;
        Sakura::Callback<void()> onShowBenchmarks;
        Sakura::Callback<void()> onReloadPlugins;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    Config config;
    SettingsHeader header;
    Sakura::Toggle timingToggle;
    OptionStepper logLevelStepper;
    SectionGroup diagnosticsGroup;
    CompactButton benchmarksButton;
    CompactButton reloadButton;
    SectionGroup toolsGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_DEVELOPER_HH
