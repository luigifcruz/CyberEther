#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_STACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_STACK_HH

#include "base.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphConfigFieldStack {
    using Stack = Sakura::NodeContentStack;

    void update(std::vector<FlowgraphConfigFieldConfig> configs, Sakura::Callback<void()> onOverflow = {});
    std::optional<Extent2D<F32>> content(Extent2D<F32> inner, F32 minimumHeight) const;
    bool hasFlexible() const;
    void appendItems(std::vector<Stack::Item>& items);

 private:
    std::vector<std::unique_ptr<Sakura::NodeFieldGrid>> grids;
    std::vector<std::unique_ptr<FlowgraphConfigFieldInstance>> fields;
    std::vector<std::optional<Extent2D<F32>>> overflows;
    std::vector<std::string> kinds;
    Sakura::Callback<void()> onOverflow;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_STACK_HH
