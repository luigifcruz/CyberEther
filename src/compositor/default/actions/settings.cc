#include "settings.hh"

namespace Jetstream {

SettingsActions::SettingsActions(DefaultCompositorState& state,
                                 DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

void SettingsActions::Select(DefaultCompositorState& state, SettingsSection section) {
    state.settings.section = section;
    ResetPages(state);
}

void SettingsActions::ResetPages(DefaultCompositorState& state) {
    state.settings.registryPage = RegistryPage::Overview;
    state.settings.registryPlugin.clear();
    state.settings.intelligencePage = IntelligencePage::Overview;
}

Result SettingsActions::handle(const MailSetBackgroundParticles& msg) {
    state.interface.backgroundParticles = msg.value;

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.interface.backgroundParticles = msg.value;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetShowTips& msg) {
    state.interface.showTips = msg.value;

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.interface.showTips = msg.value;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetTouchpadScrollSpeed& msg) {
    state.interface.touchpadScrollSpeed = msg.value;
    if (state.system.render) {
        state.system.render->setTouchpadScrollSpeed(msg.value);
    }

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.interface.touchpadScrollSpeed = msg.value;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetGraphicsScale& msg) {
    state.graphics.scale = msg.value;
    state.system.render->setScale(msg.value);

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.graphics.scale = msg.value;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetGraphicsDevice& msg) {
    state.graphics.device = msg.value == DeviceType::None
        ? std::optional<DeviceType>{}
        : std::optional<DeviceType>{msg.value};

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.graphics.device = state.graphics.device;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetSettingsSection& msg) {
    state.settings.section = msg.section;
    ResetPages(state);
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetIntelligencePage& msg) {
    state.settings.intelligencePage = msg.page;
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetRegistryPage& msg) {
    state.settings.registryPage = msg.page;
    state.settings.registryPlugin = msg.plugin;
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetDebugTimingEnabled& msg) {
    state.debug.timingEnabled = msg.value;

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.developer.timingEnabled = msg.value;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetDebugLogLevel& msg) {
    state.debug.logLevel = msg.value;
    JST_LOG_SET_DEBUG_LEVEL(state.debug.logLevel);

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.developer.logLevel = state.debug.logLevel;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailCheckForUpdates&) {
    callbacks.checkForUpdates();
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailDownloadUpdate&) {
    callbacks.downloadUpdate();
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailApplyUpdate&) {
    if (callbacks.applyUpdate()) {
        return state.system.instance->requestStop();
    }
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetPythonRuntimePath& msg) {
    state.runtime.pythonPath = msg.value;
    state.runtime.pythonValidation = PythonRuntimeContext::ValidateRuntimePath(msg.value);

    if (!state.runtime.pythonValidation.valid) {
        return Result::SUCCESS;
    }

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.runtime.python.path = msg.value;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailSetRuntimeDependencyPolicy& msg) {
    if (msg.value != "prompt" && msg.value != "allow" && msg.value != "deny") {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Ignoring invalid runtime dependency policy '{}'.",
                 msg.value);
        return Result::SUCCESS;
    }

    JST_CHECK(SetPythonDependencyPolicy(msg.value));
    state.runtime.dependencyPolicy = msg.value;

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.runtime.dependencyPolicy = msg.value;
    }));

    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailOpenPythonEnvironmentCache&) {
    std::string cacheValue;
    if (Platform::CachePath(cacheValue) != Result::SUCCESS) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "Failed to locate the Python environment cache.");
        return Result::SUCCESS;
    }

    const auto path = Platform::PathFromUtf8(cacheValue) /
                      "python-environments";
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec || Platform::OpenFolder(Platform::PathToUtf8(path)) !=
                  Result::SUCCESS) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "Failed to open the Python environment cache.");
    }
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailAddPluginPath& msg) {
    if (msg.path.empty()) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot add plugin because the path is empty.");
        return Result::SUCCESS;
    }

    if (Platform::PathFromUtf8(msg.path).extension() != ".cep") {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Plugins must be .cep bundles.");
        return Result::SUCCESS;
    }

    if (!std::filesystem::exists(Platform::PathFromUtf8(msg.path))) {
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
    JST_CHECK(Settings::SetAsync(settings));
    state.registry.stale = true;

    Select(state, SettingsSection::Registry);
    callbacks.notify(Sakura::ToastType::Success, 3000, "Plugin loaded.");
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailRemovePluginPath& msg) {
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

    JST_CHECK(Settings::SetAsync(settings));
    state.registry.stale = true;
    if (state.settings.registryPlugin == msg.path) {
        state.settings.registryPage = RegistryPage::Overview;
        state.settings.registryPlugin.clear();
    }
    callbacks.notify(Sakura::ToastType::Success, 5000, "Plugin removed. Restart CyberEther to unload registered blocks.");
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailReloadPlugin& msg) {
    if (msg.path.empty()) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot reload plugin because the path is empty.");
        return Result::SUCCESS;
    }

    if (Platform::PathFromUtf8(msg.path).extension() != ".cep") {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Plugins must be .cep bundles.");
        return Result::SUCCESS;
    }

    if (!std::filesystem::exists(Platform::PathFromUtf8(msg.path))) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "The selected plugin does not exist.");
        return Result::SUCCESS;
    }

    if (!state.flowgraph.items.empty()) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "Close open flowgraphs before reloading plugins.");
        return Result::SUCCESS;
    }

    state.registry.stale = true;
    if (Plugin::Reload(msg.path) != Result::SUCCESS) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Failed to reload plugin.");
        return Result::SUCCESS;
    }

    callbacks.notify(Sakura::ToastType::Success, 3000, "Plugin reloaded.");
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailReloadAllPlugins&) {
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
        if (Platform::PathFromUtf8(path).extension() != ".cep") {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             jst::fmt::format("Registered plugins must be .cep bundles: {}", path));
            return Result::SUCCESS;
        }
        if (!std::filesystem::exists(Platform::PathFromUtf8(path))) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             jst::fmt::format("A registered plugin does not exist: {}", path));
            return Result::SUCCESS;
        }
    }

    state.registry.stale = true;
    for (const auto& path : settings.registry.plugins) {
        if (Plugin::Reload(path) != Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Error,
                             5000,
                             jst::fmt::format("Failed to reload plugin: {}", path));
            return Result::SUCCESS;
        }
    }

    const auto count = settings.registry.plugins.size();
    callbacks.notify(Sakura::ToastType::Success,
                     3000,
                     jst::fmt::format("Reloaded modules from {} {}.", count, count == 1 ? "plugin" : "plugins"));
    return Result::SUCCESS;
}

}  // namespace Jetstream
