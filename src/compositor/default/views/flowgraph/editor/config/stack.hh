#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_STACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_STACK_HH

#include "base.hh"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphConfigFieldStack {
    using Stack = Sakura::NodeContentStack;

    void update(std::vector<FlowgraphConfigFieldConfig> configs) {
        while (fields.size() > configs.size()) {
            fields.pop_back();
        }
        while (fields.size() < configs.size()) {
            fields.push_back(std::make_unique<FlowgraphConfigFieldInstance>());
        }
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i]->update(std::move(configs[i]));
        }
    }

    bool hasFlexible() const {
        for (const auto& field : fields) {
            if (field->heightSpec().policy == FlowgraphNodeHeightPolicy::FillRemaining) {
                return true;
            }
        }
        return false;
    }

    void appendItems(std::vector<Stack::Item>& items) {
        U64 gridCount = 0;
        for (U64 i = 0; i < fields.size();) {
            if (fields[i]->isSimple()) {
                std::vector<Sakura::NodeFieldGrid::Item> gridItems;
                do {
                    gridItems.push_back({.child = &fields[i]->component()});
                    ++i;
                } while (i < fields.size() && fields[i]->isSimple());
                if (grids.size() <= gridCount) {
                    grids.push_back(std::make_unique<Sakura::NodeFieldGrid>());
                }
                auto& grid = *grids[gridCount++];
                grid.update({.items = std::move(gridItems)});
                items.push_back({
                    .child = &grid,
                    .gap = Sakura::NodeField::Gap,
                });
                continue;
            }

            const auto spec = fields[i]->heightSpec();
            std::optional<Stack::Flex> flex;
            if (spec.policy == FlowgraphNodeHeightPolicy::FillRemaining) {
                flex = Stack::Flex{
                    .minimum = spec.minimum,
                    .grow = spec.grow,
                };
            }
            items.push_back({
                .child = &fields[i]->component(),
                .flex = flex,
                .gap = Sakura::NodeField::Gap,
            });
            ++i;
        }
        while (grids.size() > gridCount) {
            grids.pop_back();
        }
    }

 private:
    std::vector<std::unique_ptr<Sakura::NodeFieldGrid>> grids;
    std::vector<std::unique_ptr<FlowgraphConfigFieldInstance>> fields;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_STACK_HH
