#include <jetstream/render/sakura/components/node/field_grid.hh>

#include <jetstream/render/sakura/components/node/field.hh>

#include "../../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kFieldColumnMinWidth = 140.0f;

}  // namespace

struct NodeFieldGrid::Impl {
    Config config;

    struct Placement {
        Component* child = nullptr;
        Rect rect;
    };

    std::vector<Placement> arrange(NodeFieldGrid& self, const Context& ctx, F32 width, F32& height) {
        std::vector<Placement> placements;
        height = 0.0f;
        std::vector<NodeFieldGrid::Item> items;
        for (const auto& item : config.items) {
            if (!item.child || item.child->visible()) {
                items.push_back(item);
            }
        }
        if (items.empty()) {
            return placements;
        }

        const F32 gap = Px(ctx, NodeField::Gap);
        const F32 minColumnWidth = Px(ctx, kFieldColumnMinWidth);
        const U64 columns = std::min<U64>(
            std::max<U64>(1, items.size()),
            (std::isfinite(width) && width > 0.0f && minColumnWidth > 0.0f)
                ? std::max<U64>(1, static_cast<U64>((width + gap) / (minColumnWidth + gap)))
                : 1);

        F32 y = 0.0f;
        bool first = true;
        const auto measure = [&](Component* child, F32 w) {
            if (!child || !child->visible()) {
                return 0.0f;
            }
            return self.measureChild(*child, ctx, {w, std::numeric_limits<F32>::infinity()}).y;
        };

        for (U64 i = 0; i < items.size();) {
            if (!first) {
                y += gap;
            }
            first = false;

            if (columns == 1 || items[i].fullWidth) {
                const F32 h = measure(items[i].child, width);
                placements.push_back({items[i].child, {0.0f, y, width, h}});
                y += h;
                ++i;
                continue;
            }

            const U64 groupStart = i;
            while (i < items.size() && !items[i].fullWidth) {
                ++i;
            }
            const U64 groupEnd = i;
            const U64 groupColumns = std::min(columns, groupEnd - groupStart);
            const F32 columnWidth = (width - static_cast<F32>(groupColumns - 1) * gap) / static_cast<F32>(groupColumns);

            F32 rowY = y;
            F32 rowHeight = 0.0f;
            std::vector<U64> rowItems;
            const auto flushRow = [&]() {
                for (const U64 index : rowItems) {
                    placements[index].rect.height = rowHeight;
                }
                rowItems.clear();
            };
            for (U64 j = groupStart; j < groupEnd; ++j) {
                const U64 column = (j - groupStart) % groupColumns;
                if (column == 0 && j != groupStart) {
                    flushRow();
                    rowY += rowHeight + gap;
                    rowHeight = 0.0f;
                }
                const F32 x = static_cast<F32>(column) * (columnWidth + gap);
                const F32 h = measure(items[j].child, columnWidth);
                rowItems.push_back(placements.size());
                placements.push_back({items[j].child, {x, rowY, columnWidth, h}});
                rowHeight = std::max(rowHeight, h);
            }
            flushRow();
            y = rowY + rowHeight;
        }

        height = y;
        return placements;
    }
};

NodeFieldGrid::NodeFieldGrid() {
    this->impl = std::make_unique<Impl>();
}

NodeFieldGrid::~NodeFieldGrid() = default;

void NodeFieldGrid::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> active;
    active.reserve(config.items.size());
    for (const auto& item : config.items) {
        active.push_back(item.child);
    }
    mountChildren(active);
    impl->config = std::move(config);
}

Extent2D<F32> NodeFieldGrid::measure(const Context& ctx, Extent2D<F32> available) {
    F32 height = 0.0f;
    impl->arrange(*this, ctx, available.x, height);
    return {FillLength(available.x), FitLength(height, available.y)};
}

void NodeFieldGrid::layout(const Context& ctx) {
    const Rect bounds = frame();
    F32 height = 0.0f;
    const auto placements = impl->arrange(*this, ctx, bounds.width, height);
    for (const auto& placement : placements) {
        if (!placement.child || !placement.child->visible()) {
            continue;
        }
        const Rect rect = {
            bounds.x + placement.rect.x,
            bounds.y + placement.rect.y,
            placement.rect.width,
            placement.rect.height,
        };
        layoutChild(ctx, *placement.child, rect);
    }
}

}  // namespace Jetstream::Sakura
