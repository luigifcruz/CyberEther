#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/settings.hh"

#include <optional>
#include <tuple>

namespace Jetstream {

struct SettingsActions {
    using Filter = std::tuple<MailSetInfoPanelEnabled,
                              MailSetBackgroundParticles,
                              MailSetGraphicsScale,
                              MailSetGraphicsDevice,
                              MailSetGraphicsFramerate,
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

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.interface.infoPanelEnabled = msg.value;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailSetBackgroundParticles& msg) {
        state.interface.backgroundParticles = msg.value;

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.interface.backgroundParticles = msg.value;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailSetGraphicsScale& msg) {
        state.graphics.scale = msg.value;

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.graphics.scale = msg.value;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailSetGraphicsDevice& msg) {
        state.graphics.device = msg.value == DeviceType::None
            ? std::optional<DeviceType>{}
            : std::optional<DeviceType>{msg.value};

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.graphics.device = state.graphics.device;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailSetGraphicsFramerate& msg) {
        state.graphics.framerate = msg.value;

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.graphics.framerate = msg.value;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailSetSettingsSection& msg) {
        state.settings.section = msg.section;
        return Result::SUCCESS;
    }

    Result handle(const MailSetDebugLatencyEnabled& msg) {
        state.debug.latencyEnabled = msg.value;

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.developer.latencyEnabled = msg.value;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailSetDebugRuntimeMetricsEnabled& msg) {
        state.debug.runtimeMetricsEnabled = msg.value;

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.developer.runtimeMetricsEnabled = msg.value;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailSetDebugLogLevel& msg) {
        state.debug.logLevel = msg.value;
        JST_LOG_SET_DEBUG_LEVEL(state.debug.logLevel);

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.developer.logLevel = state.debug.logLevel;
        JST_CHECK(Settings::Set(settings));

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
