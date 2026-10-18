#include <jetstream/render/sakura/components/table.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kIndentSpacing = 20.0f;
constexpr F32 kArrowScale = 0.70f;
constexpr F32 kResizeGrab = 4.0f;
constexpr F32 kMinColumnWidth = 32.0f;

}  // namespace

struct TableBody : public Component {
    TableBody() {
        setClipsChildren(true);
    }

    void place(const Context& ctx, Component& child, Rect rect) {
        layoutChild(ctx, child, rect);
    }

    void mount(const std::vector<Component*>& children) {
        mountChildren(children);
    }

    bool dispatch(const MouseEvent& event) {
        return mouseEventChildren(event);
    }

 protected:
    void layout(const Context&) override {}
};

struct Table::Impl {
    bool built = false;
    struct Row {
        std::vector<Component*> cells;
        bool header = false;
        U64 depth = 0;
        bool expandable = false;
        bool expanded = false;
        std::string key;
    };

    Config config;
    Box headerBackground;
    TableBody body;
    Box bodyBackgrounds;
    Box bodyLines;
    Polyline arrows;
    Box frameLines;
    Scroller scroll;
    std::vector<std::unique_ptr<Text>> texts;
    U64 usedTexts = 0;
    std::vector<Row> rows;
    std::unordered_map<std::string, bool> expanded;
    std::vector<F32> rowTops;
    std::vector<F32> rowHeights;
    std::vector<F32> columnX;
    F32 scrollY = 0.0f;
    F32 contentHeight = 0.0f;
    F32 headerHeight = 0.0f;
    Rect bodyRect;
    std::vector<F32> widthOverrides;
    F32 pixelRatio = 1.0f;
    std::optional<U64> resizing;
    F32 resizeStartX = 0.0f;
    F32 resizeStartWidth = 0.0f;
    std::optional<U64> hoveredRow;

    std::optional<U64> borderAt(const Rect& bounds, F32 x, F32 y) const {
        if (!config.resizable || !bounds.contains(x, y)) {
            return std::nullopt;
        }
        for (U64 column = 0; column + 2 < columnX.size(); ++column) {
            if (std::abs(x - (bounds.x + columnX[column + 1])) <= kResizeGrab * pixelRatio) {
                return column;
            }
        }
        return std::nullopt;
    }

    Text* text(const std::string& str, bool wrapped, bool secondary) {
        if (usedTexts == texts.size()) {
            texts.push_back(std::make_unique<Text>());
        }
        auto* cell = texts[usedTexts++].get();
        cell->update({
            .str = str,
            .tone = secondary ? Text::Tone::Secondary : Text::Tone::Primary,
            .wrapped = wrapped,
        });
        return cell;
    }

    void flatten(const std::vector<Node>& nodes, U64 depth, const std::string& prefix) {
        for (const auto& node : nodes) {
            const std::string& segment = node.id.empty() ? node.label : node.id;
            const std::string key = prefix + "/" + std::to_string(segment.size()) + ":" + segment;
            const bool expandable = !node.children.empty();
            const auto it = expanded.find(key);
            const bool open = expandable && (it != expanded.end() ? it->second : node.open);
            Row row{.depth = depth, .expandable = expandable, .expanded = open, .key = key};
            row.cells.push_back(text(node.label, false, false));
            for (U64 column = 1; column < config.columns.size(); ++column) {
                const std::string value = column - 1 < node.cells.size() ? node.cells[column - 1] : std::string();
                row.cells.push_back(text(value, config.wrapped, node.secondary));
            }
            rows.push_back(std::move(row));
            if (open) {
                flatten(node.children, depth + 1, key);
            }
        }
    }

    void rebuild() {
        usedTexts = 0;
        rows.clear();
        if (config.showHeaders) {
            Row header{.header = true};
            for (const auto& column : config.columns) {
                header.cells.push_back(text(column, false, false));
            }
            rows.push_back(std::move(header));
        }
        if (!config.nodes.empty()) {
            flatten(config.nodes, 0, "");
        } else if (!config.cellRows.empty()) {
            for (const auto& cells : config.cellRows) {
                rows.push_back({.cells = cells});
            }
        } else {
            for (const auto& values : config.rows) {
                Row row;
                for (U64 column = 0; column < config.columns.size(); ++column) {
                    row.cells.push_back(text(column < values.size() ? values[column] : std::string(),
                                             config.wrapped, false));
                }
                rows.push_back(std::move(row));
            }
        }
    }

    F32 treeOffset(const Context& ctx, const Row& row) const {
        if (config.nodes.empty() || row.header) {
            return 0.0f;
        }
        return Px(ctx, kIndentSpacing) * static_cast<F32>(row.depth) + Px(ctx, Typography::FontSize) +
               2.0f * Px(ctx, kFramePaddingX);
    }

    void columns(Table& self, const Context& ctx, F32 width) {
        pixelRatio = ctx.pixelRatio;
        const U64 count = config.columns.size();
        columnX.assign(count + 1, 0.0f);
        if (count == 0) {
            return;
        }
        const F32 padX = Px(ctx, kCellPaddingX);
        const F32 infinite = std::numeric_limits<F32>::infinity();
        std::vector<F32> widths(count, 0.0f);
        std::vector<F32> weights(count, 0.0f);
        F32 fixed = 0.0f;
        F32 weightSum = 0.0f;
        widthOverrides.resize(count, 0.0f);
        for (U64 column = 0; column < count; ++column) {
            if (widthOverrides[column] > 0.0f) {
                widths[column] = Px(ctx, widthOverrides[column]);
                fixed += widths[column];
                continue;
            }
            if (column < config.fixedColumnWidths.size() && config.fixedColumnWidths[column] > 0.0f) {
                widths[column] = Px(ctx, config.fixedColumnWidths[column]);
                fixed += widths[column];
                continue;
            }
            F32 natural = 0.0f;
            for (const auto& row : rows) {
                if (column < row.cells.size() && row.cells[column]) {
                    const F32 extra = column == 0 ? treeOffset(ctx, row) : 0.0f;
                    const F32 width = self.measureChild(*row.cells[column], ctx, {infinite, infinite}).x;
                    if (std::isfinite(width)) {
                        natural = std::max(natural, width + extra);
                    }
                }
            }
            weights[column] = std::max(natural + 2.0f * padX, 1.0f);
            weightSum += weights[column];
        }
        const F32 remaining = std::max(0.0f, width - fixed);
        for (U64 column = 0; column < count; ++column) {
            if (weights[column] > 0.0f) {
                widths[column] = std::isfinite(width) ? remaining * weights[column] / std::max(weightSum, 1.0f)
                                                      : weights[column];
            }
            columnX[column + 1] = columnX[column] + widths[column];
        }
    }

    void measureRows(Table& self, const Context& ctx) {
        const F32 padX = Px(ctx, kCellPaddingX);
        const F32 padY = Px(ctx, kCellPaddingY);
        const F32 line = Px(ctx, Typography::FontSize);
        rowTops.assign(rows.size(), 0.0f);
        rowHeights.assign(rows.size(), 0.0f);
        F32 y = 0.0f;
        for (U64 index = 0; index < rows.size(); ++index) {
            const auto& row = rows[index];
            F32 height = line;
            for (U64 column = 0; column < row.cells.size() && column + 1 < columnX.size(); ++column) {
                if (!row.cells[column]) {
                    continue;
                }
                const F32 offset = column == 0 ? treeOffset(ctx, row) : 0.0f;
                const F32 width = std::max(0.0f, columnX[column + 1] - columnX[column] - 2.0f * padX - offset);
                height = std::max(height, self.measureChild(*row.cells[column], ctx,
                                                            {width, std::numeric_limits<F32>::infinity()}).y);
            }
            rowTops[index] = y;
            rowHeights[index] = height + 2.0f * padY;
            y += rowHeights[index];
        }
        contentHeight = y;
        headerHeight = !rows.empty() && rows.front().header ? rowHeights.front() : 0.0f;
    }

    F32 rowTop(const Rect& bounds, U64 index) const {
        return bounds.y + rowTops[index] - (rows[index].header ? 0.0f : scrollY);
    }

    std::optional<U64> rowAt(const Rect& bounds, F32 x, F32 y) const {
        if (!bounds.contains(x, y)) {
            return std::nullopt;
        }
        const bool inHeader = y < bounds.y + headerHeight;
        for (U64 index = 0; index < rows.size(); ++index) {
            if (rows[index].header != inHeader) {
                continue;
            }
            const F32 top = rowTop(bounds, index);
            if (y >= top && y < top + rowHeights[index]) {
                return index;
            }
        }
        return std::nullopt;
    }
};

Table::Table() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
}

Table::~Table() = default;

void Table::update(Config config) {
    if (impl->built && config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    invalidate(Dirty::Layout);
    impl->built = true;
    impl->config = std::move(config);
    rebuild();
}

void Table::rebuild() {
    impl->rebuild();

    std::vector<Component*> header = {&impl->headerBackground};
    std::vector<Component*> body = {&impl->bodyBackgrounds};
    for (const auto& row : impl->rows) {
        for (auto* cell : row.cells) {
            if (cell) {
                (row.header ? header : body).push_back(cell);
            }
        }
    }
    body.push_back(&impl->bodyLines);
    body.push_back(&impl->arrows);
    impl->body.mount(body);
    header.push_back(&impl->body);
    header.push_back(&impl->frameLines);
    header.push_back(&impl->scroll);
    mountChildren(header);
    invalidate(Dirty::Layout);
}

Extent2D<F32> Table::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    F32 width = config.size.x > 0.0f                                ? Px(ctx, config.size.x)
                : config.size.x < 0.0f && std::isfinite(available.x) ? available.x
                                                                     : std::numeric_limits<F32>::infinity();
    impl->columns(*this, ctx, width);
    if (!std::isfinite(width)) {
        width = impl->columnX.empty() ? 0.0f : impl->columnX.back();
    }
    const F32 fitted = FitLength(width, available.x);
    if (fitted != width) {
        width = fitted;
        impl->columns(*this, ctx, width);
    }
    impl->measureRows(*this, ctx);
    F32 natural = config.size.y == 0.0f ? impl->contentHeight : 0.0f;
    if (config.maxHeight > 0.0f) {
        natural = std::min(natural, Px(ctx, config.maxHeight));
    }
    return {width, ResolveLength(ctx, config.size.y, natural, available.y)};
}

void Table::layout(const Context& ctx) {
    const Rect bounds = frame();
    impl->columns(*this, ctx, bounds.width);
    impl->measureRows(*this, ctx);
    const F32 padX = Px(ctx, kCellPaddingX);
    const F32 padY = Px(ctx, kCellPaddingY);
    const F32 line = Hairline(ctx);
    const F32 headerHeight = impl->headerHeight;
    const Rect body = {bounds.x, bounds.y + headerHeight, bounds.width, std::max(0.0f, bounds.height - headerHeight)};
    impl->bodyRect = body;
    impl->scrollY = std::clamp(impl->scrollY, 0.0f, std::max(0.0f, impl->contentHeight - bounds.height));

    impl->scroll.update({
        .contentHeight = impl->contentHeight - headerHeight,
        .scrollY = impl->scrollY,
        .wheelStep = Px(ctx, kWheelStep),
        .thickness = Px(ctx, kScrollbarThickness),
        .margin = Px(ctx, kScrollbarMargin),
        .onScrollY = [this](F32 next) {
            impl->scrollY = next;
            invalidate(Dirty::Layout);
        },
    });
    layoutChild(ctx, impl->scroll, body);
    layoutChild(ctx, impl->body, body);

    const Rect bodyClip = body;
    const Rect boundsClip = bounds;
    std::vector<Box::Instance> headerBackground;
    std::vector<Box::Instance> bodyBackgrounds;
    std::vector<Box::Instance> bodyLines;
    std::vector<Polyline::Instance> arrows;
    U64 bodyIndex = 0;
    for (U64 index = 0; index < impl->rows.size(); ++index) {
        const auto& row = impl->rows[index];
        const F32 top = impl->rowTop(bounds, index);
        const F32 height = impl->rowHeights[index];
        const Rect rowRect = {bounds.x, top, bounds.width, height};
        if (row.header) {
            headerBackground.push_back({.rect = rowRect, .backgroundColor = ctx.color("table_header_bg")});
        } else {
            const char* key = bodyIndex++ % 2 == 1 ? "table_row_bg_alt" : "table_row_bg";
            bodyBackgrounds.push_back({.rect = rowRect, .backgroundColor = ctx.color(key)});
            if (!impl->config.nodes.empty() && impl->hoveredRow == index) {
                bodyBackgrounds.push_back({.rect = rowRect, .backgroundColor = ctx.color("header_hovered")});
            }
            if (index > 0) {
                bodyLines.push_back({.rect = {bounds.x, std::floor(top), bounds.width, line},
                                     .backgroundColor = ctx.color(row.header || impl->rows[index - 1].header
                                                                      ? "table_border_strong"
                                                                      : "table_border_light")});
            }
        }
        for (U64 column = 0; column < row.cells.size() && column + 1 < impl->columnX.size(); ++column) {
            auto* cell = row.cells[column];
            if (!cell) {
                continue;
            }
            const F32 offset = column == 0 ? impl->treeOffset(ctx, row) : 0.0f;
            const F32 x = bounds.x + impl->columnX[column] + padX + offset;
            const F32 width = std::max(0.0f, impl->columnX[column + 1] - impl->columnX[column] - 2.0f * padX - offset);
            const Rect cellRect = {x, top + padY, width, std::max(0.0f, height - 2.0f * padY)};
            if (row.header) {
                layoutChild(ctx, *cell, cellRect);
            } else {
                impl->body.place(ctx, *cell, cellRect);
            }
        }
        if (row.expandable) {
            const F32 size = Px(ctx, Typography::FontSize);
            const F32 r = size * 0.40f * kArrowScale;
            const Extent2D<F32> center = {bounds.x + Px(ctx, kIndentSpacing) * static_cast<F32>(row.depth) + padX +
                                              Px(ctx, kFramePaddingX) + size * 0.5f,
                                          top + padY + size * 0.5f};
            const std::vector<Extent2D<F32>> points = row.expanded
                ? std::vector<Extent2D<F32>>{{center.x, center.y + 0.75f * r},
                                             {center.x - 0.866f * r, center.y - 0.75f * r},
                                             {center.x + 0.866f * r, center.y - 0.75f * r}}
                : std::vector<Extent2D<F32>>{{center.x + 0.75f * r, center.y},
                                             {center.x - 0.75f * r, center.y - 0.866f * r},
                                             {center.x - 0.75f * r, center.y + 0.866f * r}};
            arrows.push_back({.points = points, .color = ctx.color("text_primary"), .closed = true, .filled = true});
        }
    }

    std::vector<Box::Instance> frameLines;
    const F32 bottom = std::min(bounds.bottom(), bounds.y + impl->contentHeight - impl->scrollY);
    const F32 height = std::max(0.0f, bottom - bounds.y);
    const auto light = ctx.color("table_border_light");
    const auto strong = ctx.color("table_border_strong");
    for (U64 column = 1; column + 1 < impl->columnX.size(); ++column) {
        frameLines.push_back({.rect = {std::floor(bounds.x + impl->columnX[column]), bounds.y, line, height},
                              .backgroundColor = light});
    }
    if (headerHeight > 0.0f) {
        frameLines.push_back({.rect = {bounds.x, std::floor(bounds.y + headerHeight - line), bounds.width, line},
                              .backgroundColor = strong});
    }
    frameLines.push_back({.rect = {bounds.x, bounds.y, bounds.width, line}, .backgroundColor = strong});
    frameLines.push_back({.rect = {bounds.x, std::floor(bottom - line), bounds.width, line}, .backgroundColor = strong});
    frameLines.push_back({.rect = {bounds.x, bounds.y, line, height}, .backgroundColor = strong});
    frameLines.push_back({.rect = {std::floor(bounds.right() - line), bounds.y, line, height}, .backgroundColor = strong});

    impl->headerBackground.update({.instances = std::move(headerBackground), .clip = boundsClip});
    impl->bodyBackgrounds.update({.instances = std::move(bodyBackgrounds), .clip = bodyClip});
    impl->bodyLines.update({.instances = std::move(bodyLines), .clip = bodyClip});
    impl->arrows.update({.instances = std::move(arrows), .clip = bodyClip, .fringe = Fringe(ctx), .capacity = 256});
    impl->frameLines.update({.instances = std::move(frameLines), .clip = boundsClip});
    layoutChild(ctx, impl->headerBackground, bounds);
    layoutChild(ctx, impl->frameLines, bounds);
    impl->body.place(ctx, impl->bodyBackgrounds, body);
    impl->body.place(ctx, impl->bodyLines, body);
    impl->body.place(ctx, impl->arrows, body);
}

bool Table::mouseEvent(const MouseEvent& event) {
    const Rect bounds = frame();
    if (impl->resizing.has_value()) {
        const U64 column = *impl->resizing;
        if (event.type == MouseEventType::Move) {
            const F32 ratio = std::max(1e-3f, impl->pixelRatio);
            const F32 width = std::max(kMinColumnWidth, (impl->resizeStartWidth + event.position.x - impl->resizeStartX) / ratio);
            if (impl->widthOverrides[column] != width) {
                impl->widthOverrides[column] = width;
                invalidate(Dirty::Layout);
            }
            return true;
        }
        if (event.type == MouseEventType::Release && event.button == MouseButton::Left) {
            impl->resizing.reset();
            return true;
        }
        if (event.type == MouseEventType::Cancel) {
            impl->resizing.reset();
        }
        if (event.type == MouseEventType::Click || event.type == MouseEventType::Release) {
            return true;
        }
    }
    if (event.type == MouseEventType::Click && event.button == MouseButton::Left) {
        if (const auto column = impl->borderAt(bounds, event.position.x, event.position.y); column.has_value()) {
            impl->resizing = column;
            impl->resizeStartX = event.position.x;
            impl->resizeStartWidth = impl->columnX[*column + 1] - impl->columnX[*column];
            return true;
        }
    }
    if (event.type == MouseEventType::Move || event.type == MouseEventType::Leave ||
        event.type == MouseEventType::Cancel) {
        std::optional<U64> hovered;
        if (event.type == MouseEventType::Move && !impl->config.nodes.empty()) {
            hovered = impl->rowAt(bounds, event.position.x, event.position.y);
            if (hovered.has_value() && impl->rows[*hovered].header) {
                hovered.reset();
            }
        }
        if (hovered != impl->hoveredRow) {
            impl->hoveredRow = hovered;
            invalidate(Dirty::Layout);
        }
    }
    if (event.type == MouseEventType::Scroll && impl->body.dispatch(event)) {
        return true;
    }
    if (mouseEventChildren(event)) {
        return true;
    }
    if (event.type != MouseEventType::Click || event.button != MouseButton::Left) {
        return false;
    }
    const auto index = impl->rowAt(bounds, event.position.x, event.position.y);
    if (!index.has_value() || !impl->rows[*index].expandable) {
        return false;
    }
    const auto& row = impl->rows[*index];
    impl->expanded[row.key] = !row.expanded;
    rebuild();
    return true;
}

std::optional<SurfaceCursor> Table::cursorAt(const Extent2D<F32>& point) const {
    if (impl->resizing.has_value() || impl->borderAt(frame(), point.x, point.y).has_value()) {
        return SurfaceCursor::ResizeEW;
    }
    const auto index = impl->rowAt(frame(), point.x, point.y);
    if (!index.has_value()) {
        return std::nullopt;
    }
    return impl->rows[*index].expandable ? SurfaceCursor::Hand : SurfaceCursor::Default;
}

}  // namespace Jetstream::Sakura
