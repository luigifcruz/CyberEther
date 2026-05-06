#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH

#include "jetstream/render/sakura/sakura.hh"

#include "jetstream/parser.hh"
#include "jetstream/types.hh"

#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphConfigFieldConfig {
    std::string id;
    std::string name;
    std::string label;
    std::string help;
    std::string format;
    std::string encoded;
    Parser::Map values;
    std::function<void(Parser::Map, bool)> onApply;
    std::function<void(Result, std::string)> onError;
    std::function<void(bool, std::vector<std::string>, std::function<void(std::string)>)> onBrowsePath;
};

inline F32 ConfigUnitMultiplier(const std::string& unit) {
    if (unit == "GHz") return 1e9f;
    if (unit == "MHz") return 1e6f;
    if (unit == "kHz") return 1e3f;
    return 1.0f;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH
