#include "plugin_labels.hh"

namespace Jetstream {

std::string PluginLabels::Name(const std::string& path, const std::string& name) {
    if (!name.empty()) {
        return name;
    }
    const auto stem = Platform::PathToUtf8(Platform::PathFromUtf8(path).stem());
    return stem.empty() ? path : stem;
}

}  // namespace Jetstream
