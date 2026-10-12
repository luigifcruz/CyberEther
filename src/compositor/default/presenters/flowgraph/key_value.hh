#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH

#include "jetstream/render/sakura/components/table.hh"
#include "tools/value.hh"

namespace Jetstream::FlowgraphKeyValueDetail {

using Value::AnyToString;
using Value::KeyMatches;
using Text::ToLower;

inline Sakura::Table::Node AnyToNode(const std::string& id,
                                    const std::string& label,
                                    const std::any& value);

inline std::string EntryCount(const U64 shown, const U64 total) {
    return jst::fmt::format("{} of {} entr{}", shown, total,
                            total == 1 ? "y" : "ies");
}

inline std::string CountSummary(const U64 count, const char* singular,
                                const char* plural) {
    return jst::fmt::format("{} {}", count, count == 1 ? singular : plural);
}

inline Sakura::Table::Nodes MapToNodes(const Parser::Map& map) {
    Sakura::Table::Nodes nodes;
    nodes.reserve(map.size());
    for (const auto& entry : map) {
        nodes.push_back(AnyToNode(entry.key, entry.key, entry.value));
    }
    return nodes;
}

inline Sakura::Table::Nodes SequenceToNodes(const Parser::Sequence& sequence) {
    Sakura::Table::Nodes nodes;
    nodes.reserve(sequence.size());
    for (U64 i = 0; i < sequence.size(); ++i) {
        const std::string label = jst::fmt::format("[{}]", i);
        nodes.push_back(AnyToNode(label, label, sequence[i]));
    }
    return nodes;
}

inline Sakura::Table::Node AnyToNode(const std::string& id,
                                    const std::string& label,
                                    const std::any& value) {
    if (value.has_value() && value.type() == typeid(Parser::Map)) {
        const auto& map = std::any_cast<const Parser::Map&>(value);
        return {
            .id = id,
            .label = label,
            .cells = {CountSummary(map.size(), "key", "keys")},
            .children = MapToNodes(map),
            .secondary = true,
        };
    }
    if (value.has_value() && value.type() == typeid(Parser::Sequence)) {
        const auto& sequence = std::any_cast<const Parser::Sequence&>(value);
        return {
            .id = id,
            .label = label,
            .cells = {CountSummary(sequence.size(), "item", "items")},
            .children = SequenceToNodes(sequence),
            .secondary = true,
        };
    }
    return {.id = id, .label = label, .cells = {AnyToString(value)}};
}

inline void SortNodes(Sakura::Table::Nodes& nodes) {
    std::sort(nodes.begin(), nodes.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.label < rhs.label;
    });
}

}  // namespace Jetstream::FlowgraphKeyValueDetail

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH
