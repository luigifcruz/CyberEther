#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/parser.hh"
#include "jetstream/types.hh"

#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

enum class FlowgraphNodeHeightPolicy : U8 {
    Intrinsic,
    FillRemaining,
};

struct FlowgraphNodeHeightSpec {
    FlowgraphNodeHeightPolicy policy = FlowgraphNodeHeightPolicy::Intrinsic;
    F32 minimum = 0.0f;
    F32 grow = 1.0f;
};

struct FlowgraphConfigFieldConfig {
    std::string id;
    std::string name;
    std::string label;
    std::string help;
    Parser::Map format;
    std::string status;
    Sakura::NodeCodeEditor::StatusTone statusTone = Sakura::NodeCodeEditor::StatusTone::Info;
    std::vector<std::string> consoleOutput;
    bool consoleVisible = false;
    Parser::Map values;
    std::function<void(Parser::Map, bool)> onApply;
    std::function<void(Result, std::string)> onError;
    std::function<void(bool, std::vector<std::string>, std::function<void(std::string)>)> onBrowsePath;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH
