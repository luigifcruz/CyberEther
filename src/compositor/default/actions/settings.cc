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

Result SettingsActions::handle(const MailSetDebugRenderStatsEnabled& msg) {
    state.debug.renderStatsEnabled = msg.value;

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.developer.renderStatsEnabled = msg.value;
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

Result SettingsActions::handle(const MailInstallPlugin& msg) {
    if (msg.path.empty()) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Cannot install plugin because the path is empty.");
        return Result::SUCCESS;
    }

    if (!Plugin::IsCepPath(msg.path)) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Plugins must be .cep bundles.");
        return Result::SUCCESS;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(Platform::PathFromUtf8(msg.path), ec)) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "The selected plugin does not exist.");
        return Result::SUCCESS;
    }

    if (state.benchmark.running) {
        callbacks.notify(Sakura::ToastType::Error, 5000, "Wait for the benchmark to finish before installing plugins.");
        return Result::SUCCESS;
    }

    std::string id;
    if (Plugin::Install(msg.path, id) != Result::SUCCESS) {
        callbacks.notifyResult(Result::ERROR, "");
        return Result::SUCCESS;
    }

    const auto plugins = Plugin::List();
    const auto loaded = std::any_of(plugins.begin(), plugins.end(), [&](const auto& plugin) {
        return plugin.id == id && plugin.status == Plugin::Status::Loaded;
    });
    if (!loaded && Plugin::LoadInstalled() != Result::SUCCESS) {
        callbacks.notifyResult(Result::ERROR, "");
    }

    state.registry.stale = true;
    state.settings.registryPage = RegistryPage::Plugin;
    state.settings.registryPlugin = id;

    if (!loaded && Plugin::Path(id).empty()) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "Plugin installed but could not be loaded. Check the logs for details.");
        return Result::SUCCESS;
    }

    if (loaded) {
        callbacks.notify(Sakura::ToastType::Success,
                         5000,
                         "Plugin installed. The new version loads the next time CyberEther starts.");
    } else {
        callbacks.notify(Sakura::ToastType::Success, 3000, "Plugin installed.");
    }
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailRemovePlugin& msg) {
    if (Plugin::Remove(msg.id) != Result::SUCCESS) {
        callbacks.notifyResult(Result::ERROR, "");
        return Result::SUCCESS;
    }

    state.registry.stale = true;
    if (state.settings.registryPlugin == msg.id) {
        state.settings.registryPage = RegistryPage::Overview;
        state.settings.registryPlugin.clear();
    }
    callbacks.notify(Sakura::ToastType::Success, 5000, "Plugin removed. Restart CyberEther to unload its blocks.");
    return Result::SUCCESS;
}

Result SettingsActions::handle(const MailReloadPlugin& msg) {
    if (!state.flowgraph.items.empty()) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "Close open flowgraphs before reloading plugins.");
        return Result::SUCCESS;
    }

    if (state.benchmark.running) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "Wait for the benchmark to finish before reloading plugins.");
        return Result::SUCCESS;
    }

    state.registry.stale = true;
    if (Plugin::Reload(msg.id) != Result::SUCCESS) {
        callbacks.notifyResult(Result::ERROR, "");
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

    if (state.benchmark.running) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         "Wait for the benchmark to finish before reloading plugins.");
        return Result::SUCCESS;
    }

    std::vector<std::string> ids;
    for (const auto& plugin : Plugin::List()) {
        if (plugin.temporary) {
            ids.push_back(plugin.id);
        }
    }

    if (ids.empty()) {
        callbacks.notify(Sakura::ToastType::Info, 3000, "No plugins loaded with --plugin to reload.");
        return Result::SUCCESS;
    }

    state.registry.stale = true;
    for (const auto& id : ids) {
        if (Plugin::Reload(id) != Result::SUCCESS) {
            callbacks.notifyResult(Result::ERROR, "");
            return Result::SUCCESS;
        }
    }

    const auto count = ids.size();
    callbacks.notify(Sakura::ToastType::Success,
                     3000,
                     jst::fmt::format("Reloaded {} {}.", count, count == 1 ? "plugin" : "plugins"));
    return Result::SUCCESS;
}

}  // namespace Jetstream
