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
    std::function<void(Parser::Map, bool, std::function<void(Result)>)> onApply;
    std::function<void(Result, std::string)> onError;
    std::function<void(bool, std::vector<std::string>, std::function<void(std::string)>)> onBrowsePath;
    std::optional<std::string> preview;
    std::function<void(Extent2D<F32>)> onContentOverflow;
};

inline FlowgraphConfigFieldConfig BindFlowgraphConfigField(
    FlowgraphConfigFieldConfig config) {
    const auto map = Parser::Get<std::string>(config.format, "map");
    if (map.empty()) {
        const auto target = Parser::Get<std::string>(config.format, "config");
        if (!target.empty()) {
            config.name = target;
        }
        return config;
    }
    const auto key = Parser::Get<std::string>(config.format, "key");
    auto stored = Parser::Get<Parser::Map>(config.values, map);
    config.name = key;
    config.values = stored;
    if (!stored.contains(key) && config.format.contains("default")) {
        config.values[key] = config.format.at("default");
    }
    config.onApply = [map, stored = std::move(stored), apply = config.onApply](
        Parser::Map patch, const bool silent,
        std::function<void(Result)> complete) mutable {
        for (const auto& [key, value] : patch) {
            stored[key] = value;
        }
        if (apply) {
            apply({{map, stored}}, silent, std::move(complete));
        }
    };
    return config;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TYPES_HH
