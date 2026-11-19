#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_ADVANCED_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_ADVANCED_HH

#include "types.hh"

#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

inline void ApplyAdvancedSection(std::vector<FlowgraphConfigFieldConfig>& fields,
                                 const std::string& id,
                                 const bool shown,
                                 std::function<void(bool)> onToggle) {
    static const std::string Name = "advanced";
    std::vector<FlowgraphConfigFieldConfig> regular;
    std::vector<FlowgraphConfigFieldConfig> advanced;
    for (auto& field : fields) {
        auto& target = Parser::Get<bool>(field.format, "advanced") ? advanced : regular;
        target.push_back(std::move(field));
    }
    if (!advanced.empty()) {
        regular.push_back({
            .id = id,
            .name = Name,
            .label = "Advanced",
            .help = "Show settings that most chains leave at their defaults.",
            .format = {{"type", "bool"}},
            .values = {{Name, shown}},
            .onApply = [onToggle = std::move(onToggle), shown](Parser::Map patch,
                                                               const bool,
                                                               std::function<void(Result)> onComplete) {
                if (onToggle) {
                    onToggle(Parser::Get<bool>(patch, Name, !shown));
                }
                if (onComplete) {
                    onComplete(Result::SUCCESS);
                }
            },
        });
        if (shown) {
            std::move(advanced.begin(), advanced.end(), std::back_inserter(regular));
        }
    }
    fields = std::move(regular);
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_ADVANCED_HH
