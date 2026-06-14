#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/plugin.hh"
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
                              MailSetDebugTimingEnabled,
                              MailSetDebugLogLevel,
                              MailCheckForUpdates,
                              MailDismissUpdate,
                              MailSetPythonRuntimePath,
                              MailAddPluginPath,
                              MailRemovePluginPath,
                              MailReloadPlugin,
                              MailReloadAllPlugins>;

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
        state.system.render->setScale(msg.value);

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

    Result handle(const MailSetDebugTimingEnabled& msg) {
        state.debug.timingEnabled = msg.value;

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.developer.timingEnabled = msg.value;
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

    Result handle(const MailSetPythonRuntimePath& msg) {
        state.runtime.pythonPath = msg.value;
        state.runtime.pythonValidation = PythonRuntimeContext::ValidateRuntimePath(msg.value);

        if (!state.runtime.pythonValidation.valid) {
            return Result::SUCCESS;
        }

        Settings settings;
        JST_CHECK(Settings::Get(settings));
        settings.runtime.python.path = msg.value;
        JST_CHECK(Settings::Set(settings));

        return Result::SUCCESS;
    }

    Result handle(const MailAddPluginPath& msg) {
        if (msg.path.empty()) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot add plugin because the path is empty.");
            return Result::SUCCESS;
        }

        if (!std::filesystem::exists(msg.path)) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "The selected plugin does not exist.");
            return Result::SUCCESS;
        }

        Settings settings;
        JST_CHECK(Settings::Get(settings));

        const auto duplicate = std::find_if(settings.registry.plugins.begin(),
                                            settings.registry.plugins.end(),
                                            [&](const auto& path) {
                                                return path == msg.path;
                                            });
        if (duplicate != settings.registry.plugins.end()) {
            callbacks.notify(Sakura::ToastType::Info, 3000, "Plugin is already registered.");
            return Result::SUCCESS;
        }

        if (Plugin::Load(msg.path) != Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Failed to load plugin.");
            return Result::SUCCESS;
        }

        settings.registry.plugins.push_back(msg.path);
        JST_CHECK(Settings::Set(settings));

        state.settings.section = SettingsSection::Registry;
        state.modal.content = ModalContent::Settings;
        callbacks.notify(Sakura::ToastType::Success, 3000, "Plugin loaded.");
        return Result::SUCCESS;
    }

    Result handle(const MailRemovePluginPath& msg) {
        Settings settings;
        JST_CHECK(Settings::Get(settings));

        const auto initialSize = settings.registry.plugins.size();
        settings.registry.plugins.erase(
            std::remove(settings.registry.plugins.begin(),
                        settings.registry.plugins.end(),
                        msg.path),
            settings.registry.plugins.end());

        if (settings.registry.plugins.size() == initialSize) {
            callbacks.notify(Sakura::ToastType::Info, 3000, "Plugin was not registered.");
            return Result::SUCCESS;
        }

        JST_CHECK(Settings::Set(settings));
        callbacks.notify(Sakura::ToastType::Success, 5000, "Plugin removed. Restart CyberEther to unload registered blocks.");
        return Result::SUCCESS;
    }

    Result handle(const MailReloadPlugin& msg) {
        if (msg.path.empty()) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot reload plugin because the path is empty.");
            return Result::SUCCESS;
        }

        if (!std::filesystem::exists(msg.path)) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "The selected plugin does not exist.");
            return Result::SUCCESS;
        }

        if (!state.flowgraph.items.empty()) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Close open flowgraphs before reloading plugins.");
            return Result::SUCCESS;
        }

        if (Plugin::Reload(msg.path) != Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Failed to reload plugin.");
            return Result::SUCCESS;
        }

        callbacks.notify(Sakura::ToastType::Success, 3000, "Plugin reloaded.");
        return Result::SUCCESS;
    }

    Result handle(const MailReloadAllPlugins&) {
        if (!state.flowgraph.items.empty()) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             "Close open flowgraphs before reloading plugins.");
            return Result::SUCCESS;
        }

        Settings settings;
        JST_CHECK(Settings::Get(settings));

        if (settings.registry.plugins.empty()) {
            callbacks.notify(Sakura::ToastType::Info, 3000, "No plugins registered to reload.");
            return Result::SUCCESS;
        }

        for (const auto& path : settings.registry.plugins) {
            if (path.empty()) {
                callbacks.notify(Sakura::ToastType::Error,
                                 5000,
                                 "Cannot reload plugin because a registered path is empty.");
                return Result::SUCCESS;
            }
            if (!std::filesystem::exists(path)) {
                callbacks.notify(Sakura::ToastType::Error,
                                 5000,
                                 "A registered plugin does not exist: " + path);
                return Result::SUCCESS;
            }
        }

        for (const auto& path : settings.registry.plugins) {
            if (Plugin::Reload(path) != Result::SUCCESS) {
                callbacks.notify(Sakura::ToastType::Error,
                                 5000,
                                 "Failed to reload plugin: " + path);
                return Result::SUCCESS;
            }
        }

        const auto count = settings.registry.plugins.size();
        callbacks.notify(Sakura::ToastType::Success,
                         3000,
                         "Reloaded modules from " + std::to_string(count) + (count == 1 ? " plugin." : " plugins."));
        return Result::SUCCESS;
    }

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
