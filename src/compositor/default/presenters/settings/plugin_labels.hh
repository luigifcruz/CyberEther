#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_PLUGIN_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_PLUGIN_LABELS_HH

#include "jetstream/platform.hh"

#include <string>

namespace Jetstream {

class PluginLabels {
 public:
    static std::string Name(const std::string& path, const std::string& name = {});
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_PLUGIN_LABELS_HH
