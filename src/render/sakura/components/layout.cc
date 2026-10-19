#include <jetstream/render/sakura/components/layout.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/canvas.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

struct ClipLayer : public Component {
    ClipLayer() {
        setClipsChildren(true);
    }

    void mount(const std::vector<Component*>& children) {
        mountChildren(children);
    }

    void place(const Context& ctx, Component& child, Rect rect) {
        layoutChild(ctx, child, rect);
    }

 protected:
    void layout(const Context&) override {}
};

}  // namespace

struct Panel::Impl {
    Config config;
    Box background;
};

Panel::Panel() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->background);
}

Panel::~Panel() = default;

void Panel::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool remount = config.content != impl->config.content || !hasChild(impl->background) ||
                         (config.content && !hasChild(*config.content));
    impl->config = std::move(config);
    if (!remount) {
        return;
    }
    std::vector<Component*> children = {&impl->background};
    if (impl->config.content) {
        children.push_back(impl->config.content);
    }
    mountChildren(children);
}

Extent2D<F32> Panel::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 natural = 0.0f;
    if (config.size.x == 0.0f && config.content) {
        natural = std::ceil(measureChild(*config.content, ctx, {infinite, infinite}).x);
    }
    F32 width = ResolveLength(ctx, config.size.x, natural, available.x);
    if (config.maxWidth > 0.0f) {
        width = std::min(width, Px(ctx, config.maxWidth));
    }
    natural = 0.0f;
    if (config.size.y == 0.0f && config.content) {
        natural = measureChild(*config.content, ctx, {width, infinite}).y;
    }
    return {width, ResolveLength(ctx, config.size.y, natural, available.y)};
}

void Panel::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    impl->background.update({
        .instances = {{.rect = bounds, .visible = !config.transparent && !bounds.empty(),
                       .backgroundColor = ctx.color(config.backgroundColorKey)}},
    });
    layoutChild(ctx, impl->background, bounds);
    if (config.content) {
        layoutChild(ctx, *config.content, bounds);
    }
}

struct Stack::Impl {
    Config config;

    bool active(const Item& item) const {
        return item.child && item.child->visible();
    }

    std::vector<F32> mainSizes(Stack& self, const Context& ctx, Extent2D<F32> available) const {
        std::vector<F32> sizes(config.items.size(), 0.0f);
        const bool vertical = config.axis == Axis::Vertical;
        for (U64 i = 0; i < config.items.size(); ++i) {
            const auto& item = config.items[i];
            if (!active(item)) {
                continue;
            }
            if (item.size > 0.0f) {
                sizes[i] = Px(ctx, item.size);
                continue;
            }
            const Extent2D<F32> bounds = vertical
                ? Extent2D<F32>{available.x, std::numeric_limits<F32>::infinity()}
                : Extent2D<F32>{std::numeric_limits<F32>::infinity(), available.y};
            const auto measured = self.measureChild(*item.child, ctx, bounds);
            sizes[i] = vertical ? measured.y : measured.x;
        }
        return sizes;
    }

    U64 activeCount() const {
        U64 count = 0;
        for (const auto& item : config.items) {
            count += active(item) ? 1 : 0;
        }
        return count;
    }
};

Stack::Stack() {
    impl = std::make_unique<Impl>();
}

Stack::~Stack() = default;

void Stack::update(Config config) {
    if (config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    invalidate(Dirty::Layout);
    std::vector<Component*> children;
    children.reserve(config.items.size());
    for (const auto& item : config.items) {
        if (item.child) {
            children.push_back(item.child);
        }
    }
    mountChildren(children);
    impl->config = std::move(config);
}

Extent2D<F32> Stack::measure(const Context& ctx, Extent2D<F32> available) {
    const bool vertical = impl->config.axis == Axis::Vertical;
    const auto sizes = impl->mainSizes(*this, ctx, available);
    const U64 count = impl->activeCount();
    F32 main = count > 1 ? Px(ctx, impl->config.spacing) * static_cast<F32>(count - 1) : 0.0f;
    F32 cross = 0.0f;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        const auto& item = impl->config.items[i];
        if (!impl->active(item)) {
            continue;
        }
        main += sizes[i];
        if (!vertical) {
            cross = std::max(cross, measureChild(*item.child, ctx, {sizes[i], available.y}).y);
        } else if (!std::isfinite(available.x)) {
            cross = std::max(cross, measureChild(*item.child, ctx, available).x);
        }
    }
    if (vertical) {
        return {FillLength(available.x, cross), FitLength(main, available.y)};
    }
    return FitExtent({main, cross}, available);
}

void Stack::layout(const Context& ctx) {
    const Rect bounds = frame();
    const bool vertical = impl->config.axis == Axis::Vertical;
    auto sizes = impl->mainSizes(*this, ctx, {bounds.width, bounds.height});
    const U64 count = impl->activeCount();
    const F32 spacing = Px(ctx, impl->config.spacing);

    F32 used = count > 1 ? spacing * static_cast<F32>(count - 1) : 0.0f;
    F32 grow = 0.0f;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        if (impl->active(impl->config.items[i])) {
            used += sizes[i];
            grow += impl->config.items[i].grow;
        }
    }
    const F32 extent = vertical ? bounds.height : bounds.width;
    const F32 remaining = std::max(0.0f, extent - used);
    if (grow > 0.0f) {
        for (U64 i = 0; i < impl->config.items.size(); ++i) {
            const auto& item = impl->config.items[i];
            if (impl->active(item) && item.grow > 0.0f) {
                sizes[i] += remaining * item.grow / grow;
            }
        }
    }

    F32 cursor = vertical ? bounds.y : bounds.x;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        const auto& item = impl->config.items[i];
        if (!impl->active(item)) {
            continue;
        }
        Rect rect;
        if (vertical) {
            rect = {bounds.x, cursor, bounds.width, sizes[i]};
        } else {
            F32 height = bounds.height;
            F32 y = bounds.y;
            if (impl->config.centerCross) {
                height = std::min(bounds.height, measureChild(*item.child, ctx, {sizes[i], bounds.height}).y);
                y = bounds.y + std::floor((bounds.height - height) * 0.5f);
            }
            rect = {cursor, y, sizes[i], height};
        }
        layoutChild(ctx, *item.child, rect);
        cursor += sizes[i] + spacing;
    }
}

Spacer::Spacer() = default;
Spacer::~Spacer() = default;

void Spacer::update(Config next) {
    if (next == config) {
        return;
    }
    config = next;
    invalidate(Dirty::Layout);
}

Extent2D<F32> Spacer::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 lines = static_cast<F32>(std::max<U64>(1, config.lines) - 1);
    return {FillLength(available.x), FitLength(Px(ctx, Stack::Spacing) * lines, available.y)};
}

struct Divider::Impl {
    Config config;
    Box line;

    F32 spacing() const {
        return std::max(0.0f, config.spacing);
    }
};

Divider::Divider() {
    impl = std::make_unique<Impl>();
    addChild(impl->line);
}

Divider::~Divider() = default;

void Divider::update(Config config) {
    if (config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    invalidate(Dirty::Layout);
    impl->config = config;
}

Extent2D<F32> Divider::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 thickness = impl->config.separator ? Hairline(ctx) : 0.0f;
    return {FillLength(available.x), FitLength(2.0f * Px(ctx, impl->spacing()) + thickness, available.y)};
}

void Divider::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 thickness = Hairline(ctx);
    const Rect rect = {bounds.x, std::round(bounds.y + Px(ctx, impl->spacing())), bounds.width, thickness};
    impl->line.update({
        .instances = {{.rect = rect, .visible = impl->config.separator && !bounds.empty(),
                       .backgroundColor = ctx.color("separator")}},
        .clip = bounds,
    });
    layoutChild(ctx, impl->line, bounds);
}

struct Div::Impl {
    Config config;
    Box frame;
    bool hovered = false;

    bool interactive() const {
        return static_cast<bool>(config.onClick) || static_cast<bool>(config.onDoubleClick);
    }
};

Div::Div() {
    impl = std::make_unique<Impl>();
    addChild(impl->frame);
}

Div::~Div() = default;

void Div::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children = {&impl->frame};
    if (config.content) {
        children.push_back(config.content);
    }
    setChildren(children);
    impl->config = std::move(config);
}

Extent2D<F32> Div::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 padding = Px(ctx, config.padding);
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const bool natural = config.size.x == 0.0f || (config.size.x < 0.0f && !std::isfinite(available.x));
    F32 width = natural ? infinite : ResolveLength(ctx, config.size.x, 0.0f, available.x);
    Extent2D<F32> content = {0.0f, 0.0f};
    if (config.content) {
        const F32 inner = std::isfinite(width) ? std::max(0.0f, width - 2.0f * padding) : infinite;
        content = measureChild(*config.content, ctx, {inner, infinite});
    }
    if (!std::isfinite(width)) {
        width = FitLength(content.x + 2.0f * padding, available.x);
        if (config.content && width < content.x + 2.0f * padding) {
            content = measureChild(*config.content, ctx, {std::max(0.0f, width - 2.0f * padding), infinite});
        }
    }
    return {width, ResolveLength(ctx, config.size.y, content.y + 2.0f * padding, available.y)};
}

void Div::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 padding = Px(ctx, config.padding);
    const bool highlighted = config.selected || (impl->hovered && impl->interactive());
    const auto key = config.selected ? config.selectedColorKey : highlighted ? config.hoveredColorKey : config.colorKey;
    const auto borderColor = ctx.color(config.borderColorKey);
    impl->frame.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color(key)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = config.border ? Px(ctx, config.borderWidth) : 0.0f,
        .borderColor = borderColor,
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->frame, bounds);
    if (config.content) {
        layoutChild(ctx, *config.content, {bounds.x + padding, bounds.y + padding,
                                           std::max(0.0f, bounds.width - 2.0f * padding),
                                           std::max(0.0f, bounds.height - 2.0f * padding)});
    }
}

bool Div::mouseEvent(const MouseEvent& event) {
    if (mouseEventChildren(event)) {
        return true;
    }
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != impl->hovered) {
                impl->hovered = inside;
                if (impl->interactive()) {
                    invalidate(Dirty::Layout);
                }
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered) {
                impl->hovered = false;
                if (impl->interactive()) {
                    invalidate(Dirty::Layout);
                }
            }
            return false;
        case MouseEventType::Click:
            if (!inside || event.button != MouseButton::Left || !impl->interactive()) {
                return false;
            }
            if (event.clicks >= 2 && impl->config.onDoubleClick) {
                impl->config.onDoubleClick();
            } else if (impl->config.onClick) {
                impl->config.onClick();
            }
            return true;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Div::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return impl->interactive() ? SurfaceCursor::Hand : SurfaceCursor::Default;
}

namespace {

constexpr F32 kScrollbarSize = 12.0f;
constexpr F32 kScrollbarGrabInset = 3.0f;
constexpr F32 kTabPaddingX = 12.0f;
constexpr F32 kTabPaddingY = 5.0f;
constexpr F32 kTabRounding = 10.0f;
constexpr F32 kTabSpacing = 8.0f;
constexpr F32 kTabBarBorderSize = 1.0f;

}  // namespace

struct ScrollArea::Impl {
    Config config;
    Box border;
    ClipLayer clipLayer;
    Scroller scroll;
    F32 scrollY = 0.0f;
    F32 contentHeight = 0.0f;
};

ScrollArea::ScrollArea() {
    impl = std::make_unique<Impl>();
    addChild(impl->scroll);
    addChild(impl->border);
    addChild(impl->clipLayer);
}

ScrollArea::~ScrollArea() = default;

void ScrollArea::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children;
    if (config.content) {
        children.push_back(config.content);
    }
    impl->clipLayer.mount(children);
    impl->config = std::move(config);
}

Extent2D<F32> ScrollArea::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 width = FillLength(available.x);
    F32 natural = 0.0f;
    if (config.height == 0.0f && config.content) {
        const F32 padding = config.border ? Px(ctx, kWindowPadding) : 0.0f;
        const F32 inner = std::max(0.0f, width - 2.0f * padding - (config.alwaysScrollbar ? Px(ctx, kScrollbarSize) : 0.0f));
        natural = measureChild(*config.content, ctx, {inner, std::numeric_limits<F32>::infinity()}).y + 2.0f * padding;
    }
    return {width, ResolveLength(ctx, config.height, natural, available.y)};
}

void ScrollArea::layout(const Context& ctx) {
    const Rect bounds = frame();
    const bool border = impl->config.border;
    const F32 padding = border ? Px(ctx, kWindowPadding) : 0.0f;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 scrollbar = Px(ctx, kScrollbarSize);
    F32 innerWidth = std::max(0.0f, bounds.width - 2.0f * padding - scrollbar);
    impl->contentHeight = 0.0f;
    if (impl->config.content) {
        if (!impl->config.alwaysScrollbar) {
            const F32 fullWidth = std::max(0.0f, bounds.width - 2.0f * padding);
            const F32 height = measureChild(*impl->config.content, ctx, {fullWidth, infinite}).y + 2.0f * padding;
            if (height <= bounds.height) {
                scrollbar = 0.0f;
                innerWidth = fullWidth;
            }
        }
        impl->contentHeight = measureChild(*impl->config.content, ctx, {innerWidth, infinite}).y + 2.0f * padding;
    }
    impl->scrollY = std::clamp(impl->scrollY, 0.0f, std::max(0.0f, impl->contentHeight - bounds.height));

    impl->border.update({
        .instances = {{.rect = bounds, .visible = border && !bounds.empty(), .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
        .clip = bounds,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("border"),
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->border, bounds);

    impl->scroll.update({
        .contentHeight = impl->contentHeight,
        .scrollY = impl->scrollY,
        .wheelStep = Px(ctx, kWheelStep),
        .thickness = scrollbar,
        .margin = 0.0f,
        .thumbInset = Px(ctx, kScrollbarGrabInset),
        .trackColorKey = "scrollbar_bg",
        .thumbColorKey = "scrollbar_grab",
        .onScrollY = [this](F32 next) {
            impl->scrollY = next;
            invalidate(Dirty::Layout);
        },
    });
    layoutChild(ctx, impl->scroll, bounds);

    const F32 inset = padding * 0.5f;
    const Rect clipRect = {bounds.x + inset, bounds.y + inset,
                           std::max(0.0f, bounds.width - scrollbar - 2.0f * inset),
                           std::max(0.0f, bounds.height - 2.0f * inset)};
    layoutChild(ctx, impl->clipLayer, clipRect);
    if (impl->config.content) {
        impl->clipLayer.place(ctx, *impl->config.content,
                              {bounds.x + padding, bounds.y + padding - impl->scrollY, innerWidth,
                               std::max(0.0f, impl->contentHeight - 2.0f * padding)});
    }
}

struct TabBar::Impl {
    Config config;
    Box tabs;
    Box separator;
    std::vector<std::unique_ptr<Text>> labels;
    U64 active = 0;
    std::optional<U64> hovered;
    std::vector<Rect> tabRects;

    std::optional<U64> tabAt(F32 x, F32 y) const {
        for (U64 index = 0; index < tabRects.size(); ++index) {
            if (tabRects[index].contains(x, y)) {
                return index;
            }
        }
        return std::nullopt;
    }

    F32 textHeight(TabBar& self, const Context& ctx) const {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        F32 height = config.labels.empty() ? Px(ctx, Typography::FontSize) : 0.0f;
        for (U64 index = 0; index < config.labels.size(); ++index) {
            height = std::max(height, self.measureChild(*labels[index], ctx, {infinite, infinite}).y);
        }
        return height;
    }

    F32 barHeight(TabBar& self, const Context& ctx) const {
        return textHeight(self, ctx) + 2.0f * Px(ctx, kTabPaddingY);
    }

    std::vector<Component*> mounted;

    Component* page() const {
        return active < config.pages.size() ? config.pages[active] : nullptr;
    }

    void showActive() {
        for (U64 index = 0; index < config.pages.size(); ++index) {
            if (config.pages[index]) {
                config.pages[index]->setVisible(index == active);
            }
        }
    }
};

TabBar::TabBar() {
    impl = std::make_unique<Impl>();
}

TabBar::~TabBar() = default;

void TabBar::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    while (impl->labels.size() < impl->config.labels.size()) {
        impl->labels.push_back(std::make_unique<Text>());
    }
    if (impl->active >= impl->config.labels.size()) {
        impl->active = 0;
    }
    std::vector<Component*> children = {&impl->tabs, &impl->separator};
    for (U64 index = 0; index < impl->config.labels.size(); ++index) {
        impl->labels[index]->update({.str = impl->config.labels[index]});
        children.push_back(impl->labels[index].get());
    }
    for (auto* page : impl->config.pages) {
        if (page) {
            children.push_back(page);
        }
    }
    const bool owned = std::all_of(children.begin(), children.end(), [this](const Component* child) {
        return hasChild(*child);
    });
    if (children != impl->mounted || !owned) {
        mountChildren(children);
        impl->mounted = std::move(children);
    }
    impl->showActive();
}

Extent2D<F32> TabBar::measure(const Context& ctx, Extent2D<F32> available) {
    F32 height = impl->barHeight(*this, ctx);
    if (auto* page = impl->page()) {
        height += Px(ctx, Stack::Spacing) + measureChild(*page, ctx, {available.x, std::numeric_limits<F32>::infinity()}).y;
    }
    return {FillLength(available.x), FitLength(height, available.y)};
}

void TabBar::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 textHeight = impl->textHeight(*this, ctx);
    const F32 barHeight = textHeight + 2.0f * Px(ctx, kTabPaddingY);
    const F32 padX = Px(ctx, kTabPaddingX);
    const F32 padY = Px(ctx, kTabPaddingY);
    const F32 infinite = std::numeric_limits<F32>::infinity();

    std::vector<Box::Instance> tabs;
    impl->tabRects.clear();
    F32 x = bounds.x;
    for (U64 index = 0; index < impl->config.labels.size(); ++index) {
        auto& label = *impl->labels[index];
        const F32 textWidth = measureChild(label, ctx, {infinite, infinite}).x;
        const Rect tab = {x, bounds.y, textWidth + 2.0f * padX, barHeight};
        impl->tabRects.push_back(tab);
        const char* key = index == impl->active ? "tab_selected"
                        : impl->hovered == index ? "tab_hovered"
                        : "tab";
        tabs.push_back({
            .rect = {tab.x, tab.y + Px(ctx, 1.0f), tab.width, tab.height - Px(ctx, 1.0f)},
            .visible = !tab.empty(),
            .backgroundColor = ctx.color(key),
            .corners = Box::TopLeft | Box::TopRight,
        });
        layoutChild(ctx, label, {tab.x + padX, tab.y + padY, textWidth, textHeight});
        x += tab.width + Px(ctx, kTabSpacing);
    }
    const F32 rounding = impl->tabRects.empty()
        ? 0.0f
        : std::min(Px(ctx, kTabRounding), impl->tabRects.front().width * 0.5f - 1.0f);
    impl->tabs.update({.instances = std::move(tabs), .clip = bounds, .cornerRadius = std::max(0.0f, rounding)});
    layoutChild(ctx, impl->tabs, bounds);

    const F32 line = Px(ctx, kTabBarBorderSize);
    impl->separator.update({
        .instances = {{.rect = {bounds.x, bounds.y + barHeight - line, bounds.width, line},
                       .visible = !bounds.empty(),
                       .backgroundColor = ctx.color("tab_selected")}},
        .clip = bounds,
    });
    layoutChild(ctx, impl->separator, bounds);

    if (auto* page = impl->page()) {
        const F32 top = bounds.y + barHeight + Px(ctx, Stack::Spacing);
        layoutChild(ctx, *page, {bounds.x, top, bounds.width, std::max(0.0f, bounds.bottom() - top)});
    }
}

bool TabBar::mouseEvent(const MouseEvent& event) {
    const auto tab = impl->tabAt(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (tab != impl->hovered) {
                impl->hovered = tab;
                invalidate(Dirty::Layout);
            }
            break;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered.has_value()) {
                impl->hovered.reset();
                invalidate(Dirty::Layout);
            }
            break;
        case MouseEventType::Click:
            if (tab.has_value() && event.button == MouseButton::Left) {
                if (*tab != impl->active) {
                    impl->active = *tab;
                    impl->showActive();
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            break;
        default:
            break;
    }
    return mouseEventChildren(event);
}

std::optional<SurfaceCursor> TabBar::cursorAt(const Extent2D<F32>& point) const {
    if (impl->tabAt(point.x, point.y).has_value()) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

struct SplitView::Impl {
    Config config;
    Box line;
};

SplitView::SplitView() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
}

SplitView::~SplitView() = default;

void SplitView::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool remount = config.left != impl->config.left || config.right != impl->config.right ||
                         !hasChild(impl->line) || (config.left && !hasChild(*config.left)) ||
                         (config.right && !hasChild(*config.right));
    if (remount) {
        std::vector<Component*> children = {&impl->line};
        if (config.left) {
            children.push_back(config.left);
        }
        if (config.right) {
            children.push_back(config.right);
        }
        mountChildren(children);
    }
    impl->config = std::move(config);
}

Extent2D<F32> SplitView::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 width = FillLength(available.x);
    F32 natural = 0.0f;
    if (config.height == 0.0f) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        const F32 padX = Px(ctx, kCellPaddingX);
        const F32 leftWidth = std::min(Px(ctx, config.leftWidth), width);
        const F32 rightWidth = std::max(0.0f, width - leftWidth - 2.0f * padX - Hairline(ctx));
        if (config.left) {
            natural = std::max(natural, measureChild(*config.left, ctx, {leftWidth, infinite}).y);
        }
        if (config.right) {
            natural = std::max(natural, measureChild(*config.right, ctx, {rightWidth, infinite}).y);
        }
        natural += Px(ctx, kCellPaddingY);
    }
    return {width, ResolveLength(ctx, config.height, natural, available.y)};
}

void SplitView::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 padX = Px(ctx, kCellPaddingX);
    const F32 padY = Px(ctx, kCellPaddingY);
    const F32 leftWidth = std::min(Px(ctx, impl->config.leftWidth), bounds.width);
    const F32 line = Hairline(ctx);
    const F32 lineX = std::floor(bounds.x + leftWidth + padX);
    impl->line.update({
        .instances = {{.rect = {lineX, bounds.y, line, bounds.height},
                       .visible = !bounds.empty(),
                       .backgroundColor = ctx.color("table_border_light")}},
        .clip = bounds,
    });
    layoutChild(ctx, impl->line, bounds);
    const F32 innerHeight = std::max(0.0f, bounds.height - padY);
    if (impl->config.left) {
        layoutChild(ctx, *impl->config.left, {bounds.x, bounds.y + padY, leftWidth, innerHeight});
    }
    if (impl->config.right) {
        const F32 rightX = lineX + line + padX;
        layoutChild(ctx, *impl->config.right,
                    {rightX, bounds.y + padY, std::max(0.0f, bounds.right() - rightX), innerHeight});
    }
}

struct Grid::Impl {
    Config config;
    ClipLayer clipLayer;
    Scroller scroll;
    F32 scrollY = 0.0f;
    std::vector<F32> rowHeights;

    U64 columns() const {
        return std::max<U64>(1, config.columns);
    }

    F32 cellWidth(const Context& ctx, F32 width) const {
        const F32 padX = Px(ctx, config.cellPadding.x);
        const F32 outer = config.outerPadding ? padX : 0.0f;
        const F32 n = static_cast<F32>(columns());
        return std::max(0.0f, (width - 2.0f * outer - (n - 1.0f) * 2.0f * padX) / n);
    }

    template<typename Measure>
    F32 measureRows(const Context& ctx, F32 width, const Measure& measure) {
        const F32 padY = Px(ctx, config.cellPadding.y);
        const F32 cell = cellWidth(ctx, width);
        const U64 n = columns();
        rowHeights.assign((config.items.size() + n - 1) / n, 0.0f);
        for (U64 index = 0; index < config.items.size(); ++index) {
            auto* item = config.items[index];
            if (!item) {
                continue;
            }
            const F32 height = measure(*item, cell);
            rowHeights[index / n] = std::max(rowHeights[index / n], height + 2.0f * padY);
        }
        F32 total = 0.0f;
        for (const F32 height : rowHeights) {
            total += height;
        }
        return total;
    }
};

Grid::Grid() {
    impl = std::make_unique<Impl>();
    addChild(impl->scroll);
    addChild(impl->clipLayer);
}

Grid::~Grid() = default;

void Grid::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> items;
    for (auto* item : config.items) {
        if (item) {
            items.push_back(item);
        }
    }
    impl->clipLayer.mount(items);
    impl->config = std::move(config);
}

Extent2D<F32> Grid::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 natural = 0.0f;
    if (config.size.x == 0.0f) {
        for (auto* item : config.items) {
            if (item) {
                natural = std::max(natural, measureChild(*item, ctx, {infinite, infinite}).x);
            }
        }
        const F32 padX = Px(ctx, config.cellPadding.x);
        const F32 n = static_cast<F32>(impl->columns());
        natural = n * natural + (n - 1.0f) * 2.0f * padX + (config.outerPadding ? 2.0f * padX : 0.0f);
    }
    const F32 width = ResolveLength(ctx, config.size.x, natural, available.x);
    const auto measureItem = [&](Component& item, F32 cell) {
        return measureChild(item, ctx, {cell, infinite}).y;
    };
    natural = config.size.y == 0.0f ? impl->measureRows(ctx, width, measureItem) : 0.0f;
    return {width, ResolveLength(ctx, config.size.y, natural, available.y)};
}

void Grid::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    const F32 padX = Px(ctx, config.cellPadding.x);
    const F32 padY = Px(ctx, config.cellPadding.y);
    const F32 outer = config.outerPadding ? padX : 0.0f;
    const auto measureItem = [&](Component& item, F32 cell) {
        return measureChild(item, ctx, {cell, std::numeric_limits<F32>::infinity()}).y;
    };
    F32 width = bounds.width;
    F32 contentHeight = impl->measureRows(ctx, width, measureItem);
    const F32 scrollbar = Px(ctx, kScrollbarSize);
    if (contentHeight > bounds.height + 0.5f) {
        width = std::max(0.0f, bounds.width - scrollbar);
        contentHeight = impl->measureRows(ctx, width, measureItem);
    }
    impl->scrollY = std::clamp(impl->scrollY, 0.0f, std::max(0.0f, contentHeight - bounds.height));

    impl->scroll.update({
        .contentHeight = contentHeight,
        .scrollY = impl->scrollY,
        .wheelStep = Px(ctx, kWheelStep),
        .thickness = scrollbar,
        .margin = 0.0f,
        .thumbInset = Px(ctx, kScrollbarGrabInset),
        .trackColorKey = "scrollbar_bg",
        .thumbColorKey = "scrollbar_grab",
        .onScrollY = [this](F32 next) {
            impl->scrollY = next;
            invalidate(Dirty::Layout);
        },
    });
    layoutChild(ctx, impl->scroll, bounds);
    layoutChild(ctx, impl->clipLayer, bounds);

    const U64 n = impl->columns();
    const F32 cell = impl->cellWidth(ctx, width);
    F32 y = bounds.y - impl->scrollY;
    for (U64 row = 0; row < impl->rowHeights.size(); ++row) {
        for (U64 column = 0; column < n; ++column) {
            const U64 index = row * n + column;
            if (index >= config.items.size() || !config.items[index]) {
                continue;
            }
            const F32 x = bounds.x + outer + static_cast<F32>(column) * (cell + 2.0f * padX);
            impl->clipLayer.place(ctx, *config.items[index],
                                  {x, y + padY, cell, std::max(0.0f, impl->rowHeights[row] - 2.0f * padY)});
        }
        y += impl->rowHeights[row];
    }
}

}  // namespace Jetstream::Sakura
