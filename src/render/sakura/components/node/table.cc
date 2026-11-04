#include <jetstream/render/sakura/components/node/table.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr U64 kCellCharacters = 128;

}  // namespace

struct NodeTable::Impl {
    Config config;
    Box backgrounds;
    Box borders;
    std::vector<std::unique_ptr<Label>> columns;
    TextMetrics metrics;
    std::vector<F32> weights;
    F32 weightsFontSize = -1.0f;

    U64 rowCount() const {
        return config.rows.size() + (config.showHeaders ? 1 : 0);
    }

    F32 rowHeight(const Context& ctx) const {
        return Px(ctx, NodeFontOf(ctx) + 8.0f);
    }
};

NodeTable::NodeTable() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->backgrounds);
    addChild(impl->borders);
}

NodeTable::~NodeTable() = default;

void NodeTable::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    if (config.columns != impl->config.columns || config.rows != impl->config.rows ||
        config.showHeaders != impl->config.showHeaders) {
        impl->weightsFontSize = -1.0f;
    }
    impl->config = std::move(config);
    if (impl->columns.size() != impl->config.columns.size()) {
        impl->columns.resize(impl->config.columns.size());
        std::vector<Component*> children = {&impl->backgrounds, &impl->borders};
        for (auto& column : impl->columns) {
            if (!column) {
                column = std::make_unique<Label>();
            }
            children.push_back(column.get());
        }
        setChildren(children);
    }
}

Extent2D<F32> NodeTable::measure(const Context& ctx, Extent2D<F32> available) {
    if (impl->config.columns.empty()) {
        return {FillLength(available.x), FitLength(0.0f, available.y)};
    }
    return {FillLength(available.x), FitLength(impl->rowHeight(ctx) * static_cast<F32>(impl->rowCount()), available.y)};
}

void NodeTable::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    const U64 columns = config.columns.size();
    const F32 rowHeight = impl->rowHeight(ctx);
    const U64 rowCapacity = static_cast<U64>(std::ceil(static_cast<F32>(ctx.framebufferSize.y) /
                                                       std::max(1.0f, rowHeight))) + 2;
    const U64 backgroundCapacity = rowCapacity;
    const U64 borderCapacity = rowCapacity + columns + 4;
    impl->metrics.setWindow(ctx.render);

    if (columns == 0 || bounds.empty()) {
        impl->backgrounds.update({
            .instances = {},
            .capacity = backgroundCapacity,
        });
        impl->borders.update({.instances = {}, .capacity = borderCapacity});
        for (auto& column : impl->columns) {
            column->update({
                .instances = {},
                .fontName = Typography::MonoFont,
                .maxCharacters = kCellCharacters,
                .capacity = rowCapacity,
            });
        }
        layoutChild(ctx, impl->backgrounds, bounds);
        layoutChild(ctx, impl->borders, bounds);
        for (auto& column : impl->columns) {
            layoutChild(ctx, *column, bounds);
        }
        return;
    }

    const F32 fontSize = Px(ctx, NodeFontOf(ctx));
    const F32 cellPadX = Px(ctx, kNodeFramePadX);
    const F32 line = Hairline(ctx);

    if (impl->weightsFontSize != fontSize || impl->weights.size() != columns) {
        impl->weightsFontSize = fontSize;
        impl->weights.assign(columns, 1.0f);
        for (U64 c = 0; c < columns; ++c) {
            F32 width = config.showHeaders ? impl->metrics.measure(Typography::MonoFont, config.columns[c], fontSize) : 0.0f;
            for (const auto& row : config.rows) {
                if (c < row.size()) {
                    width = std::max(width, impl->metrics.measure(Typography::MonoFont, row[c], fontSize));
                }
            }
            impl->weights[c] = std::max(width, fontSize);
        }
    }
    const auto& weights = impl->weights;
    F32 weightSum = 0.0f;
    for (const F32 weight : weights) {
        weightSum += weight;
    }
    std::vector<F32> columnX(columns + 1, bounds.x);
    for (U64 c = 0; c < columns; ++c) {
        columnX[c + 1] = columnX[c] + bounds.width * weights[c] / std::max(1e-3f, weightSum);
    }
    columnX[columns] = bounds.right();

    std::vector<Box::Instance> backgrounds;
    std::vector<Box::Instance> borders;
    std::vector<std::vector<Label::Instance>> cells(columns);

    const auto addRow = [&](U64 index, F32 y, const std::vector<std::string>& values, bool header) {
        const Rect rowRect = {bounds.x, y, bounds.width, rowHeight};
        if (header) {
            backgrounds.push_back({.rect = Snap(rowRect), .backgroundColor = ctx.color("table_header_bg")});
        } else if (index % 2 == 1) {
            backgrounds.push_back({.rect = Snap(rowRect), .backgroundColor = ctx.color("table_row_bg_alt")});
        } else {
            backgrounds.push_back({.rect = Snap(rowRect), .backgroundColor = ctx.color("table_row_bg")});
        }
        for (U64 c = 0; c < columns; ++c) {
            const Rect cellRect = {columnX[c] + cellPadX, y, std::max(0.0f, columnX[c + 1] - columnX[c] - 2.0f * cellPadX),
                                   rowHeight};
            cells[c].push_back({
                .rect = cellRect,
                .str = c < values.size() ? values[c] : std::string(),
                .color = ctx.color("text_primary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            });
        }
    };

    const U64 rowCount = impl->rowCount();
    const F32 y = bounds.y + rowHeight * static_cast<F32>(rowCount);
    U64 firstRow = 0;
    U64 lastRow = 0;
    if (!clipRect.empty()) {
        firstRow = std::min<U64>(rowCount, static_cast<U64>(std::max(0.0f, (clipRect.y - bounds.y) / rowHeight)));
        lastRow = std::min<U64>(rowCount, static_cast<U64>(std::ceil(std::max(0.0f, (clipRect.bottom() - bounds.y) /
                                                                                       rowHeight))));
        lastRow = std::min(lastRow, firstRow + rowCapacity);
    }
    const U64 headerRows = config.showHeaders ? 1 : 0;
    for (U64 r = firstRow; r < lastRow; ++r) {
        const F32 rowY = bounds.y + rowHeight * static_cast<F32>(r);
        if (r < headerRows) {
            addRow(0, rowY, config.columns, true);
        } else {
            addRow(r - headerRows, rowY, config.rows[r - headerRows], false);
        }
    }

    const auto light = ctx.color("table_border_light");
    const auto strong = ctx.color("table_border_strong");
    for (U64 r = std::max<U64>(1, firstRow); r < std::min(rowCount, lastRow + 1); ++r) {
        const F32 lineY = std::floor(bounds.y + rowHeight * static_cast<F32>(r));
        borders.push_back({.rect = {bounds.x, lineY, bounds.width, line},
                           .backgroundColor = (config.showHeaders && r == 1) ? strong : light});
    }
    for (U64 c = 1; c < columns; ++c) {
        borders.push_back({.rect = {std::floor(columnX[c]), bounds.y, line, y - bounds.y}, .backgroundColor = light});
    }
    borders.push_back({.rect = {bounds.x, bounds.y, bounds.width, line}, .backgroundColor = strong});
    borders.push_back({.rect = {bounds.x, std::floor(y - line), bounds.width, line}, .backgroundColor = strong});
    borders.push_back({.rect = {bounds.x, bounds.y, line, y - bounds.y}, .backgroundColor = strong});
    borders.push_back({.rect = {std::floor(bounds.right() - line), bounds.y, line, y - bounds.y}, .backgroundColor = strong});

    impl->backgrounds.update({
        .instances = std::move(backgrounds),
        .clip = bounds,
        .capacity = backgroundCapacity,
    });
    impl->borders.update({
        .instances = std::move(borders),
        .clip = bounds,
        .capacity = borderCapacity,
    });
    for (U64 c = 0; c < columns; ++c) {
        const Rect columnRect = {columnX[c], bounds.y, columnX[c + 1] - columnX[c], bounds.height};
        impl->columns[c]->update({
            .instances = std::move(cells[c]),
            .clip = Intersect(bounds, columnRect),
            .fontName = Typography::MonoFont,
            .maxCharacters = kCellCharacters,
            .capacity = rowCapacity,
        });
    }

    layoutChild(ctx, impl->backgrounds, bounds);
    layoutChild(ctx, impl->borders, bounds);
    for (auto& column : impl->columns) {
        layoutChild(ctx, *column, bounds);
    }
}

}  // namespace Jetstream::Sakura
