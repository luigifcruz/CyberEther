#include <jetstream/render/sakura/components/dropdown.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/access.hh"
#include "../detail/popup_layer.hh"
#include "../detail/text_metrics.hh"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr U64 kRowSlots = 24;
constexpr U64 kMaxCharacters = 128;

}  // namespace

struct DropdownBody : public Component {
    struct Metrics {
        F32 fontSize = Typography::FontSize;
        F32 padX = 0.0f;
        F32 padY = 0.0f;
        F32 gap = 0.0f;
        F32 rounding = 0.0f;
        F32 marginX = 0.0f;
        F32 marginY = 0.0f;
        F32 panelRounding = 0.0f;
        F32 borderSize = 0.0f;

        bool operator==(const Metrics&) const = default;
    };

    Box panel;
    Box highlights;
    Label texts;
    Scroller scroll;
    TextMetrics textMetrics;
    F32 anchorWidth = 0.0f;
    U64 visibleRows = 8;
    std::function<void(U64)> onActivate;

    std::vector<std::string> options;
    std::optional<U64> selected;
    Metrics metrics;
    F32 scrollY = 0.0f;
    bool revealSelected = false;
    std::optional<U64> hovered;
    std::optional<U64> pressed;
    Rect content;
    Rect viewport;
    F32 rowHeight = 0.0f;
    F32 stride = 0.0f;

    DropdownBody() {
        setClipsChildren(true);
        addChild(panel);
        addChild(highlights);
        addChild(texts);
        addChild(scroll);
    }

    void refresh() {
        invalidate(Dirty::Layout);
    }

    std::optional<U64> rowAt(const Extent2D<F32>& point) const {
        if (stride <= 0.0f || !viewport.contains(point.x, point.y)) {
            return std::nullopt;
        }
        const F32 offset = point.y - content.y + scrollY;
        if (offset < 0.0f) {
            return std::nullopt;
        }
        const U64 row = static_cast<U64>(offset / stride);
        if (row >= options.size() || offset - static_cast<F32>(row) * stride > rowHeight) {
            return std::nullopt;
        }
        return row;
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        const F32 unit = ctx.pixelRatio;
        textMetrics.setWindow(ctx.render);
        F32 labelWidth = 0.0f;
        for (const auto& option : options) {
            labelWidth = std::max(labelWidth, textMetrics.measure(Typography::MonoFont, option, metrics.fontSize * unit));
        }
        const U64 count = options.size();
        const F32 rowHeight = (metrics.fontSize + 2.0f * metrics.padY) * unit;
        const F32 gap = metrics.gap * unit;
        const F32 contentHeight = count > 0 ? static_cast<F32>(count) * (rowHeight + gap) - gap : 0.0f;
        const F32 maxHeight = visibleRows > 0 ? static_cast<F32>(visibleRows) * (rowHeight + gap) - gap : contentHeight;
        const F32 marginX = metrics.marginX * unit;
        const F32 marginY = metrics.marginY * unit;
        const F32 innerWidth = std::max(anchorWidth - 2.0f * marginX, labelWidth + 2.0f * metrics.padX * unit);
        return FitExtent({std::ceil(innerWidth + 2.0f * marginX), std::ceil(std::min(contentHeight, maxHeight) + 2.0f * marginY)},
                         available);
    }

    void layout(const Context& ctx) override {
        const Rect outer = frame();
        const F32 unit = ctx.pixelRatio;
        const F32 marginX = metrics.marginX * unit;
        const F32 marginY = metrics.marginY * unit;
        const Rect bounds = {outer.x + marginX, outer.y + marginY,
                             std::max(0.0f, outer.width - 2.0f * marginX), std::max(0.0f, outer.height - 2.0f * marginY)};

        panel.update({
            .instances = {{.rect = outer, .visible = !outer.empty(), .backgroundColor = ctx.color("popup_bg")}},
            .cornerRadius = metrics.panelRounding * unit,
            .borderWidth = metrics.borderSize > 0.0f ? unit : 0.0f,
            .borderColor = ctx.color("border"),
            .strokeFringe = true,
        });
        layoutChild(ctx, panel, outer);

        const F32 fontSize = metrics.fontSize * unit;
        const F32 padX = metrics.padX * unit;
        const F32 padY = metrics.padY * unit;
        const F32 gap = metrics.gap * unit;
        rowHeight = fontSize + 2.0f * padY;
        stride = rowHeight + gap;
        const U64 count = options.size();
        const F32 contentHeight = count > 0 ? static_cast<F32>(count) * stride - gap : 0.0f;

        if (revealSelected && selected.has_value() && bounds.height > 0.0f) {
            revealSelected = false;
            const F32 bottom = static_cast<F32>(*selected) * stride + rowHeight;
            scrollY = std::max(0.0f, bottom - bounds.height);
        }
        scrollY = std::clamp(scrollY, 0.0f, std::max(0.0f, contentHeight - bounds.height));

        scroll.update({
            .id = "dropdown:scroll",
            .contentHeight = contentHeight,
            .scrollY = scrollY,
            .wheelStep = Px(ctx, kWheelStep),
            .thickness = Px(ctx, kScrollbarThickness),
            .margin = Px(ctx, kScrollbarMargin),
            .onLayout = [this](Rect contentRect, std::optional<Rect> clipRect) {
                content = contentRect;
                viewport = clipRect.value_or(contentRect);
            },
            .onScrollY = [this](F32 next) {
                scrollY = next;
                invalidate(Dirty::Layout);
            },
        });
        layoutChild(ctx, scroll, bounds);

        std::vector<Box::Instance> boxes;
        std::vector<Label::Instance> labels;
        const U64 first = stride > 0.0f ? static_cast<U64>(std::max(0.0f, std::floor(scrollY / stride))) : 0;
        for (U64 row = first; row < count && labels.size() < kRowSlots; ++row) {
            const F32 top = content.y + static_cast<F32>(row) * stride - scrollY;
            if (top >= viewport.bottom()) {
                break;
            }
            const Rect rowRect = {content.x, top, content.width, rowHeight};
            const bool isHovered = hovered == row;
            const bool isSelected = selected == row;
            if (isHovered || isSelected) {
                const char* key = isHovered && pressed == row ? "header_active"
                                  : isHovered                ? "header_hovered"
                                                             : "header";
                const F32 y0 = std::round(rowRect.y);
                const F32 y1 = std::round(rowRect.bottom());
                boxes.push_back({.rect = {std::round(rowRect.x), y0, std::round(rowRect.width), y1 - y0},
                                 .backgroundColor = ctx.color(key)});
            }
            labels.push_back({
                .rect = {rowRect.x + padX, rowRect.y + padY, std::max(0.0f, rowRect.width - padX), fontSize},
                .str = options[row],
                .color = ctx.color("text_primary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            });
        }

        highlights.update({
            .instances = std::move(boxes),
            .clip = viewport,
            .cornerRadius = metrics.rounding * unit,
            .capacity = kRowSlots,
        });
        texts.update({
            .instances = std::move(labels),
            .clip = viewport,
            .fontName = Typography::MonoFont,
            .maxCharacters = kMaxCharacters,
            .capacity = kRowSlots,
        });
        layoutChild(ctx, highlights, bounds);
        layoutChild(ctx, texts, bounds);
    }

    bool mouseEvent(const MouseEvent& event) override {
        if (mouseEventChildren(event)) {
            if (event.type == MouseEventType::Scroll) {
                hovered = rowAt(event.position);
            }
            return true;
        }
        switch (event.type) {
            case MouseEventType::Move: {
                const auto row = rowAt(event.position);
                if (row != hovered) {
                    hovered = row;
                    invalidate(Dirty::Layout);
                }
                return row.has_value();
            }
            case MouseEventType::Click: {
                if (event.button != MouseButton::Left) {
                    return false;
                }
                pressed = rowAt(event.position);
                hovered = pressed;
                invalidate(Dirty::Layout);
                return pressed.has_value();
            }
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left || !pressed.has_value()) {
                    return false;
                }
                const auto row = rowAt(event.position);
                const bool activate = row == pressed && row.has_value();
                pressed.reset();
                invalidate(Dirty::Layout);
                if (activate && onActivate) {
                    onActivate(*row);
                }
                return true;
            }
            case MouseEventType::Leave:
            case MouseEventType::Cancel: {
                hovered.reset();
                pressed.reset();
                invalidate(Dirty::Layout);
                return false;
            }
            default:
                return false;
        }
    }
};

struct Dropdown::Impl {
    Config config;
    DropdownBody body;
    Component* owner = nullptr;
    std::weak_ptr<bool> ownerAlive;
    bool open = false;

    void notify() const {
        if (owner && !ownerAlive.expired()) {
            Access::invalidate(*owner);
        }
    }

    Impl() {
        body.onActivate = [this](U64 row) {
            open = false;
            notify();
            if (row < config.options.size() && config.onSelect) {
                config.onSelect(row);
            }
        };
    }
};

Dropdown::Dropdown() {
    impl = std::make_unique<Impl>();
}

Dropdown::~Dropdown() = default;

void Dropdown::open() {
    impl->open = true;
    impl->body.scrollY = 0.0f;
    impl->body.revealSelected = true;
    impl->body.hovered.reset();
    impl->body.pressed.reset();
    impl->body.refresh();
    impl->notify();
}

void Dropdown::close() {
    impl->open = false;
    impl->notify();
}

bool Dropdown::isOpen() const {
    return impl->open;
}

void Dropdown::update(Config config) {
    impl->config = std::move(config);
    if (impl->body.options != impl->config.options) {
        impl->body.options = impl->config.options;
        impl->body.refresh();
    }
    const bool stateChanged = impl->body.selected != impl->config.selected ||
                              impl->body.visibleRows != impl->config.visibleRows;
    impl->body.selected = impl->config.selected;
    impl->body.visibleRows = impl->config.visibleRows;
    const F32 padX = impl->config.rowPadding.x;
    const F32 padY = impl->config.rowPadding.y;
    const F32 margin = std::max(impl->config.padding, 1.0f);
    const DropdownBody::Metrics metrics = {
        .fontSize = Typography::FontSize,
        .padX = padX,
        .padY = padY,
        .gap = impl->config.rowGap,
        .rounding = kFrameRounding,
        .marginX = margin,
        .marginY = margin,
        .panelRounding = kPopupRounding,
        .borderSize = 1.0f,
    };
    if (stateChanged || impl->body.metrics != metrics) {
        impl->body.metrics = metrics;
        impl->body.refresh();
    }
}

void Dropdown::request(const Context& ctx, const Rect& anchor) {
    impl->owner = LayoutOwner();
    impl->ownerAlive = impl->owner ? Access::alive(*impl->owner) : std::weak_ptr<bool>();
    if (!impl->open) {
        return;
    }
    if (impl->body.anchorWidth != anchor.width) {
        impl->body.anchorWidth = anchor.width;
        impl->body.refresh();
    }
    RequestPopup(ctx, {
        .content = &impl->body,
        .anchor = anchor,
        .placement = PopupPlacement::Below,
        .onDismiss = [impl = impl.get()]() {
            impl->open = false;
            impl->notify();
        },
    });
}

}  // namespace Jetstream::Sakura
