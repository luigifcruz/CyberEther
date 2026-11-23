#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_PLUGINS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_PLUGINS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/flowgraph.hh"
#include "jetstream/logger.hh"
#include "jetstream/plugin.hh"

#include "plugin_gallery.hh"
#include "tools/text.hh"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

namespace Jetstream {

struct PluginActions {
    using Clock = std::chrono::steady_clock;
    using Filter = std::tuple<MailInstallGalleryPlugins, MailOpenPluginDetails, MailClosePluginDetails>;

    static constexpr auto RetryInterval = std::chrono::seconds(30);
    static constexpr auto RefreshInterval = std::chrono::hours(1);

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;
    Plugin::Gallery::Source source;

    PluginActions(DefaultCompositorState& state,
                  DefaultCompositorCallbacks& callbacks,
                  Plugin::Gallery::Source source = Plugin::Gallery::Source::Default());

    static std::vector<std::string> Missing(const std::vector<char>& blob);
    static std::vector<std::string> MissingInFile(const std::string& path);
    static void Request(DefaultCompositorState& state,
                        std::vector<std::string> missing,
                        std::string path,
                        std::vector<char> blob);

    void reconcile(Clock::time_point now = Clock::now());
    void shutdown();
    Result handle(const MailInstallGalleryPlugins& msg);
    Result handle(const MailOpenPluginDetails& msg);
    Result handle(const MailClosePluginDetails& msg);

 private:
    void reconcileUpdate();
    void reconcileCatalog(Clock::time_point now);
    void reconcileInstall();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_PLUGINS_HH
