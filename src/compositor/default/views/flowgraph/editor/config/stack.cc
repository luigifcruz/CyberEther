#include "stack.hh"

namespace Jetstream {

void FlowgraphConfigFieldStack::update(std::vector<FlowgraphConfigFieldConfig> configs, Sakura::Callback<void()> onOverflow) {
    this->onOverflow = std::move(onOverflow);
    while (fields.size() > configs.size()) {
        fields.pop_back();
    }
    while (fields.size() < configs.size()) {
        fields.push_back(std::make_unique<FlowgraphConfigFieldInstance>());
    }
    overflows.resize(fields.size());
    kinds.resize(fields.size());
    for (U64 i = 0; i < fields.size(); ++i) {
        auto kind = Parser::Get<std::string>(configs[i].format, "type");
        if (kind != kinds[i]) {
            kinds[i] = std::move(kind);
            overflows[i].reset();
        }
        configs[i].onContentOverflow = [this, i](Extent2D<F32> overflow) {
            if (overflows[i] == overflow) {
                return;
            }
            overflows[i] = overflow;
            if (this->onOverflow) {
                this->onOverflow();
            }
        };
        fields[i]->update(std::move(configs[i]));
    }
}

std::optional<Extent2D<F32>> FlowgraphConfigFieldStack::content(Extent2D<F32> inner, F32 minimumHeight) const {
    std::optional<Extent2D<F32>> content;
    for (const auto& overflow : overflows) {
        if (!overflow.has_value()) {
            continue;
        }
        if (!content.has_value()) {
            content = inner;
        }
        content->x = std::max(content->x, inner.x + overflow->x);
        content->y += overflow->y;
    }
    if (content.has_value()) {
        content->y = std::max(content->y, minimumHeight);
    }
    return content;
}

bool FlowgraphConfigFieldStack::hasFlexible() const {
    for (const auto& field : fields) {
        if (field->heightSpec().policy == FlowgraphNodeHeightPolicy::FillRemaining) {
            return true;
        }
    }
    return false;
}

void FlowgraphConfigFieldStack::appendItems(std::vector<Stack::Item>& items) {
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

}  // namespace Jetstream
