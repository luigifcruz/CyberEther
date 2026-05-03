#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"

#include <tuple>

namespace Jetstream {

struct SettingsActions {
    using Filter = std::tuple<MailSetInfoPanelEnabled,
                              MailSetBackgroundParticles,
                              MailSetSettingsSection,
                              MailSetDebugLatencyEnabled,
                              MailSetDebugRuntimeMetricsEnabled,
                              MailSetDebugLogLevel,
                              MailCheckForUpdates,
                              MailDismissUpdate>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    Result handle(const MailSetInfoPanelEnabled& msg) {
        state.interface.infoPanelEnabled = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetBackgroundParticles& msg) {
        state.interface.backgroundParticles = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetSettingsSection& msg) {
        state.settings.section = msg.section;
        return Result::SUCCESS;
    }

    Result handle(const MailSetDebugLatencyEnabled& msg) {
        state.debug.latencyEnabled = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetDebugRuntimeMetricsEnabled& msg) {
        state.debug.runtimeMetricsEnabled = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetDebugLogLevel& msg) {
        state.debug.logLevel = msg.value;
        JST_LOG_SET_DEBUG_LEVEL(state.debug.logLevel);
        return Result::SUCCESS;
    }

    Result handle(const MailCheckForUpdates&) {
        state.update.checking = true;
        return Result::SUCCESS;
    }

    Result handle(const MailDismissUpdate&) {
        state.update.available = false;
        return Result::SUCCESS;
    }

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
