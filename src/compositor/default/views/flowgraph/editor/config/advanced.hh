#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_ADVANCED_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_ADVANCED_HH

#include "types.hh"

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

void ApplyAdvancedSection(std::vector<FlowgraphConfigFieldConfig>& fields,
                          const std::string& id,
                          bool shown,
                          Sakura::Callback<void(bool)> onToggle);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_ADVANCED_HH
