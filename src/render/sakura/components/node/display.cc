#include <jetstream/render/sakura/components/node/display.hh>

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/tools/icons_fa.hh>

#include "common.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

using namespace NodeStyle;

namespace {

constexpr U64 kCellCharacters = 128;

}  // namespace

struct NodeLabel::Impl {
    Config config;
    Box frame;
    Label text;
    TextMetrics metrics;
};

NodeLabel::NodeLabel() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->text);
}

NodeLabel::~NodeLabel() = default;

void NodeLabel::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (impl->config.boxed) {
        setChildren({&impl->frame, &impl->text});
    } else {
        setChildren({&impl->text});
    }
}

Extent2D<F32> NodeLabel::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 line = Px(ctx, FontOf(ctx));
    return {FillLength(available.x), FitLength(impl->config.boxed ? line + Px(ctx, 16.0f) : line, available.y)};
}

void NodeLabel::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 fontSize = Px(ctx, FontOf(ctx));
    const F32 padding = config.boxed ? std::min(Px(ctx, 8.0f), bounds.width * 0.5f) : 0.0f;
    const F32 paddingY = config.boxed ? Px(ctx, 8.0f) : 0.0f;
    const Rect textRect = {bounds.x + padding, bounds.y + paddingY, std::max(0.0f, bounds.width - 2.0f * padding), fontSize};

    if (config.boxed) {
        impl->frame.update({
            .instances = {{
                .rect = Snap(bounds),
                .visible = !bounds.empty(),
                .backgroundColor = ctx.color("card"),
            }},
            .clip = bounds,
            .cornerRadius = Px(ctx, 6.0f),
            .borderWidth = Hairline(ctx),
            .borderColor = ctx.color("table_border_strong"),
        });
    }

    impl->metrics.setWindow(ctx.render);
    std::string line = config.str.substr(0, config.str.find('\n'));
    const bool overflows = config.align != Align::Left &&
                           impl->metrics.measure(MonoFont, line, fontSize) > textRect.width;
    const I32 alignX = overflows                     ? 0
                       : config.align == Align::Center ? 1
                       : config.align == Align::Right  ? 2
                                                       : 0;
    impl->text.update({
        .instances = {{
            .rect = textRect,
            .str = std::move(line),
            .visible = !bounds.empty(),
            .color = ctx.color(config.colorKey),
            .fontSize = fontSize,
            .alignment = {alignX, 1},
        }},
        .clip = Intersect(bounds, textRect),
        .fontName = MonoFont,
        .maxCharacters = 256,
    });

    if (config.boxed) {
        layoutChild(ctx, impl->frame, bounds);
    }
    layoutChild(ctx, impl->text, bounds);
}

struct NodeProgressBar::Impl {
    Config config;
    Box frame;
    Box fill;
    Label overlay;
    TextMetrics metrics;
};

NodeProgressBar::NodeProgressBar() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->frame);
    addChild(impl->fill);
    addChild(impl->overlay);
}

NodeProgressBar::~NodeProgressBar() = default;

void NodeProgressBar::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> NodeProgressBar::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FontOf(ctx) + 10.0f), available.y)};
}

void NodeProgressBar::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 fraction = std::clamp(config.value, 0.0f, 1.0f);
    const F32 rounding = Px(ctx, FrameRounding);
    const F32 fontSize = Px(ctx, FontOf(ctx));
    impl->metrics.setWindow(ctx.render);

    impl->frame.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty(), .backgroundColor = ctx.color("card")}},
        .clip = bounds,
        .cornerRadius = rounding,
    });

    const F32 fillEnd = bounds.x + bounds.width * fraction;
    impl->fill.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty() && fraction > 0.0f,
                       .backgroundColor = ctx.color(config.colorKey)}},
        .clip = Intersect(bounds, {bounds.x, bounds.y, std::max(0.0f, fillEnd - bounds.x), bounds.height}),
        .cornerRadius = rounding,
    });

    const F32 textWidth = impl->metrics.measure(MonoFont, config.overlay, fontSize);
    const F32 textX = std::clamp(fillEnd + Px(ctx, ItemSpacing),
                                 bounds.x,
                                 std::max(bounds.x, bounds.right() - textWidth - Px(ctx, 8.0f)));
    impl->overlay.update({
        .instances = {{
            .rect = {textX, bounds.y, textWidth, bounds.height},
            .str = config.overlay,
            .visible = !bounds.empty() && !config.overlay.empty(),
            .color = ctx.color("text_primary"),
            .fontSize = fontSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = MonoFont,
        .maxCharacters = 256,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->fill, bounds);
    layoutChild(ctx, impl->overlay, bounds);
}

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
        return Px(ctx, FontOf(ctx) + 8.0f);
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
                .fontName = MonoFont,
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

    const F32 fontSize = Px(ctx, FontOf(ctx));
    const F32 cellPadX = Px(ctx, FramePadX);
    const F32 line = Hairline(ctx);

    if (impl->weightsFontSize != fontSize || impl->weights.size() != columns) {
        impl->weightsFontSize = fontSize;
        impl->weights.assign(columns, 1.0f);
        for (U64 c = 0; c < columns; ++c) {
            F32 width = config.showHeaders ? impl->metrics.measure(MonoFont, config.columns[c], fontSize) : 0.0f;
            for (const auto& row : config.rows) {
                if (c < row.size()) {
                    width = std::max(width, impl->metrics.measure(MonoFont, row[c], fontSize));
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
            .fontName = MonoFont,
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

struct NodeSurface::Impl {
    Config config;
    SurfaceImage surface;
    Box button;
    Text icon;
    Rect buttonRect;
    Pointer pointer;
    bool hovered = false;
    bool buttonHovered = false;
};

NodeSurface::NodeSurface() {
    this->impl = std::make_shared<Impl>();
    addChild(impl->surface);
    addChild(impl->button);
    addChild(impl->icon);
}

NodeSurface::~NodeSurface() = default;

void NodeSurface::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->surface.update({
        .texture = impl->config.texture,
        .rounding = kFrameRounding,
        .onSize = impl->config.onSize,
    });
    impl->icon.update({
        .str = ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER,
        .colorKey = "action_btn_text",
    });
}

Extent2D<F32> NodeSurface::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, impl->config.minimumHeight), available.y)};
}

void NodeSurface::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    impl->hovered = impl->pointer.inside(clipRect);

    const F32 buttonSize = Px(ctx, 24.0f);
    const F32 buttonPadding = Px(ctx, 8.0f);
    impl->buttonRect = {bounds.right() - buttonSize - buttonPadding, bounds.y + buttonPadding, buttonSize, buttonSize};
    impl->buttonHovered = impl->config.detachOverlay && impl->hovered && impl->pointer.inside(impl->buttonRect);

    layoutChild(ctx, impl->surface, bounds);

    const bool showButton = impl->config.detachOverlay && impl->hovered;
    impl->button.update({
        .instances = {{
            .rect = impl->buttonRect,
            .visible = showButton,
            .backgroundColor = ctx.color(impl->buttonHovered ? "node_overlay_button_hovered" : "node_overlay_button"),
        }},
        .clip = bounds,
        .cornerRadius = Px(ctx, 4.0f),
    });
    layoutChild(ctx, impl->button, bounds);

    impl->icon.setVisible(showButton);
    const auto iconSize = measureChild(impl->icon, ctx, {std::numeric_limits<F32>::infinity(), std::numeric_limits<F32>::infinity()});
    layoutChild(ctx, impl->icon, {
        impl->buttonRect.x + std::floor((impl->buttonRect.width - iconSize.x) * 0.5f),
        impl->buttonRect.y + std::floor((impl->buttonRect.height - iconSize.y) * 0.5f),
        iconSize.x,
        iconSize.y,
    });
}

bool NodeSurface::mouseEvent(const MouseEvent& event) {
    if (impl->pointer.track(event, Intersect(frame(), clip()), {frame(), impl->buttonRect})) {
        invalidate(Dirty::Layout);
    }
    if (!impl->config.detachOverlay || event.type != MouseEventType::Click || event.button != MouseButton::Left) {
        return false;
    }
    if (!impl->hovered || !impl->buttonRect.contains(event.position.x, event.position.y)) {
        return false;
    }
    if (impl->config.onDetach) {
        impl->config.onDetach();
    }
    return true;
}

}  // namespace Jetstream::Sakura
