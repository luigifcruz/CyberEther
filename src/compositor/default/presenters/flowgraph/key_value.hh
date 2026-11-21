#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH

#include "jetstream/render/sakura/components/table.hh"
#include "tools/value.hh"

namespace Jetstream::FlowgraphKeyValueDetail {

using Value::AnyToString;
using Value::KeyMatches;

Sakura::Table::Node AnyToNode(const std::string& id,
                              const std::string& label,
                              const std::any& value);

std::string EntryCount(const U64 shown, const U64 total);

void SortNodes(Sakura::Table::Nodes& nodes);

}  // namespace Jetstream::FlowgraphKeyValueDetail

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH
