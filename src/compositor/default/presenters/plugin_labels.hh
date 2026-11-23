#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_PLUGIN_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_PLUGIN_LABELS_HH

#include "jetstream/logger.hh"
#include "jetstream/plugin.hh"
#include "plugin_gallery.hh"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace Jetstream {

class PluginLabels {
 public:
    static bool Installed(const std::vector<Plugin::Info>& installed, const Plugin::Gallery::Entry& entry);
    static std::string Category(const Plugin::Gallery::Entry& entry, bool installed);
    static std::string Block(const std::string& plugin, const std::string& block);
    static std::vector<std::string> Blocks(const Plugin::Gallery::Entry& entry);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_PLUGIN_LABELS_HH
