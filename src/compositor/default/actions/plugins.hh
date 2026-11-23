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
    using Filter = std::tuple<MailInstallGalleryPlugins>;

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

    void reconcile();
    void shutdown();
    Result handle(const MailInstallGalleryPlugins& msg);

 private:
    void reconcileUpdate();
    void reconcileInstall();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_PLUGINS_HH
