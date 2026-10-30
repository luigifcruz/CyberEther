#include <jetstream/render/sakura/components/tab_bar.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kTabPaddingX = 12.0f;
constexpr F32 kTabPaddingY = 5.0f;
constexpr F32 kTabRounding = 10.0f;
constexpr F32 kTabSpacing = 8.0f;
constexpr F32 kTabBarBorderSize = 1.0f;

}  // namespace

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

}  // namespace Jetstream::Sakura
