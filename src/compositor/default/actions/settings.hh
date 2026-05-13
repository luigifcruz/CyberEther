#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/registry.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <filesystem>
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
                              MailDismissUpdate,
                              MailAddRegistryLibraryPath,
                              MailRemoveRegistryLibraryPath>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    SettingsActions(DefaultCompositorState& state,
                    DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

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

    Result handle(const MailAddRegistryLibraryPath& msg) {
        if (msg.path.empty()) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot add registry library because the path is empty.");
            return Result::SUCCESS;
        }

        if (!std::filesystem::exists(msg.path)) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "The selected registry library does not exist.");
            return Result::SUCCESS;
        }

        Settings settings;
        JST_CHECK(Settings::Get(settings));

        const auto duplicate = std::find_if(settings.registry.dynamicLibraries.begin(),
                                            settings.registry.dynamicLibraries.end(),
                                            [&](const auto& path) {
                                                return path == msg.path;
                                            });
        if (duplicate != settings.registry.dynamicLibraries.end()) {
            callbacks.notify(Sakura::ToastType::Info, 3000, "Registry library is already registered.");
            return Result::SUCCESS;
        }

        if (Registry::LoadDynamicLibrary(msg.path) != Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Failed to load registry library.");
            return Result::SUCCESS;
        }

        settings.registry.dynamicLibraries.push_back(msg.path);
        JST_CHECK(Settings::Set(settings));

        state.settings.section = SettingsSection::Registry;
        state.modal.content = ModalContent::Settings;
        callbacks.notify(Sakura::ToastType::Success, 3000, "Registry library loaded.");
        return Result::SUCCESS;
    }

    Result handle(const MailRemoveRegistryLibraryPath& msg) {
        Settings settings;
        JST_CHECK(Settings::Get(settings));

        const auto initialSize = settings.registry.dynamicLibraries.size();
        settings.registry.dynamicLibraries.erase(
            std::remove(settings.registry.dynamicLibraries.begin(),
                        settings.registry.dynamicLibraries.end(),
                        msg.path),
            settings.registry.dynamicLibraries.end());

        if (settings.registry.dynamicLibraries.size() == initialSize) {
            callbacks.notify(Sakura::ToastType::Info, 3000, "Registry library was not registered.");
            return Result::SUCCESS;
        }

        JST_CHECK(Settings::Set(settings));
        callbacks.notify(Sakura::ToastType::Success, 5000, "Registry library removed. Restart CyberEther to unload registered blocks.");
        return Result::SUCCESS;
    }

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
