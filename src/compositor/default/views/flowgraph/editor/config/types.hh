#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/parser.hh"
#include "jetstream/types.hh"

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
    Sakura::StatusTone statusTone = Sakura::StatusTone::Info;
    std::vector<std::string> consoleOutput;
    bool consoleVisible = false;
    Parser::Map values;
    Sakura::Callback<void(Parser::Map, bool, Sakura::Callback<void(Result)>)> onApply;
    Sakura::Callback<void(Result, std::string)> onError;
    Sakura::Callback<void(bool, std::vector<std::string>, Sakura::Callback<void(std::string)>)> onBrowsePath;
    std::optional<std::string> preview;
    std::optional<std::string> draft;
    Sakura::Callback<void(std::optional<std::string>, std::string)> onDraft;
    Sakura::Callback<void(Extent2D<F32>)> onContentOverflow;
    Sakura::Callback<void(const std::string&)> onOpenUrl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH
