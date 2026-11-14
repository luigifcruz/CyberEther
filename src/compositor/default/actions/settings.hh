#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH

#include "persist.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"
#include "jetstream/settings.hh"
#include "runtime/python/dependencies/coordinator.hh"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace Jetstream {

struct SettingsActions {
    using Filter = std::tuple<MailSetBackgroundParticles,
                              MailSetShowTips,
                              MailSetTouchpadScrollSpeed,
                              MailSetGraphicsScale,
                              MailSetGraphicsDevice,
                              MailSetSettingsSection,
                              MailSetRegistryPage,
                              MailSetIntelligencePage,
                              MailSetDebugTimingEnabled,
                              MailSetDebugLogLevel,
                              MailCheckForUpdates,
                              MailDownloadUpdate,
                              MailApplyUpdate,
                              MailSetPythonRuntimePath,
                              MailSetRuntimeDependencyPolicy,
                              MailOpenPythonEnvironmentCache,
                              MailAddPluginPath,
                              MailRemovePluginPath,
                              MailReloadPlugin,
                              MailReloadAllPlugins>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    SettingsActions(DefaultCompositorState& state,
                    DefaultCompositorCallbacks& callbacks);
    static void Select(DefaultCompositorState& state, SettingsSection section);
    static void ResetPages(DefaultCompositorState& state);
    Result handle(const MailSetBackgroundParticles& msg);
    Result handle(const MailSetShowTips& msg);
    Result handle(const MailSetTouchpadScrollSpeed& msg);
    Result handle(const MailSetGraphicsScale& msg);
    Result handle(const MailSetGraphicsDevice& msg);
    Result handle(const MailSetSettingsSection& msg);
    Result handle(const MailSetIntelligencePage& msg);
    Result handle(const MailSetRegistryPage& msg);
    Result handle(const MailSetDebugTimingEnabled& msg);
    Result handle(const MailSetDebugLogLevel& msg);
    Result handle(const MailCheckForUpdates&);
    Result handle(const MailDownloadUpdate&);
    Result handle(const MailApplyUpdate&);
    Result handle(const MailSetPythonRuntimePath& msg);
    Result handle(const MailSetRuntimeDependencyPolicy& msg);
    Result handle(const MailOpenPythonEnvironmentCache&);
    Result handle(const MailAddPluginPath& msg);
    Result handle(const MailRemovePluginPath& msg);
    Result handle(const MailReloadPlugin& msg);
    Result handle(const MailReloadAllPlugins&);

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_SETTINGS_HH
