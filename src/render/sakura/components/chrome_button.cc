#include <jetstream/render/sakura/components/chrome_button.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/polyline.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct ChromeButton::Impl {
    Config config;
    Box background;
    Text label;
    Polyline glyph;
    Menu menu;
    bool hovered = false;
    bool open = false;
    bool menuStale = true;
    bool menuOpen = false;
    Rect menuBounds;

    const MenuEntry* entryAt(const std::vector<U64>& path) const {
        const std::vector<MenuEntry>* level = &config.entries;
        const MenuEntry* entry = nullptr;
        for (const U64 index : path) {
            if (index >= level->size()) {
                return nullptr;
            }
            entry = &(*level)[index];
            level = &entry->children;
        }
        return entry;
    }

    std::vector<MenuEntry> forward(const std::vector<MenuEntry>& entries, const std::vector<U64>& path) const {
        std::vector<MenuEntry> result;
        for (U64 index = 0; index < entries.size(); ++index) {
            auto next = path;
            next.push_back(index);
            MenuEntry entry = entries[index];
            if (entry.onClick) {
                entry.onClick = [this, next]() {
                    if (const auto* current = entryAt(next); current && current->onClick) {
                        current->onClick();
                    }
                };
            }
            entry.children = forward(entries[index].children, next);
            result.push_back(std::move(entry));
        }
        return result;
    }
};

ChromeButton::ChromeButton() {
    impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->label);
    addChild(impl->glyph);
    addChild(impl->menu);
}

ChromeButton::~ChromeButton() = default;

void ChromeButton::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
        impl->menuStale = true;
    }
    impl->config = std::move(config);
    if (impl->config.entries.empty()) {
        impl->open = false;
    }
    impl->label.update({
        .str = impl->config.label,
        .font = impl->config.font,
        .colorKey = impl->config.colorKey,
        .scale = impl->config.scale,
    });
}

bool ChromeButton::menuOpen() const {
    return impl->open;
}

Extent2D<F32> ChromeButton::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const auto text = measureChild(impl->label, ctx, {infinite, infinite});
    const F32 height = std::round(ResolveLength(ctx, config.height, text.y + 2.0f * Px(ctx, config.padding.y), available.y));
    if (config.glyph != Glyph::None) {
        return FitExtent({height, height}, available);
    }
    return FitExtent({std::ceil(text.x + 2.0f * Px(ctx, config.padding.x)), height}, available);
}

void ChromeButton::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    impl->background.update({
        .instances = {{.rect = bounds, .visible = impl->hovered || impl->open,
                       .backgroundColor = ctx.color(config.hoverColorKey)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
    });
    layoutChild(ctx, impl->background, bounds);

    const F32 infinite = std::numeric_limits<F32>::infinity();
    const auto size = measureChild(impl->label, ctx, {infinite, infinite});
    impl->label.setVisible(config.glyph == Glyph::None);
    layoutChild(ctx, impl->label, {bounds.x + std::floor((bounds.width - size.x) * 0.5f),
                                   bounds.y + std::floor((bounds.height - size.y) * 0.5f), size.x, size.y});

    std::vector<Polyline::Instance> marks;
    if (config.glyph == Glyph::Plus) {
        const F32 arm = std::round(bounds.height * 0.22f);
        const Extent2D<F32> center = {bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f};
        const auto color = ctx.color(config.colorKey.empty() ? "text_primary" : config.colorKey);
        const F32 thickness = Px(ctx, 1.5f);
        marks.push_back({.points = {{center.x - arm, center.y}, {center.x + arm, center.y}}, .color = color,
                         .thickness = thickness});
        marks.push_back({.points = {{center.x, center.y - arm}, {center.x, center.y + arm}}, .color = color,
                         .thickness = thickness});
    }
    impl->glyph.update({.instances = std::move(marks), .clip = bounds, .fringe = Fringe(ctx), .capacity = 64});
    layoutChild(ctx, impl->glyph, bounds);

    if (impl->menuStale || impl->menuOpen != impl->open || impl->menuBounds != bounds) {
        impl->menuStale = false;
        impl->menuOpen = impl->open;
        impl->menuBounds = bounds;
        impl->menu.update({
            .entries = impl->forward(config.entries, {}),
            .open = impl->open,
            .placement = PopupPlacement::Below,
            .atCursor = false,
            .passthrough = bounds,
            .onClose = [impl = impl.get(), this]() {
                if (impl->open) {
                    impl->open = false;
                    invalidate(Dirty::Layout);
                }
            },
        });
    }
    layoutChild(ctx, impl->menu, bounds);
}

bool ChromeButton::mouseEvent(const MouseEvent& event) {
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != impl->hovered) {
                impl->hovered = inside;
                invalidate(Dirty::Layout);
            }
            mouseEventChildren(event);
            return false;
        case MouseEventType::Click:
            if (event.button == MouseButton::Left && inside) {
                if (impl->config.entries.empty()) {
                    if (impl->config.onClick) {
                        impl->config.onClick();
                    }
                } else {
                    impl->open = !impl->open;
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            return mouseEventChildren(event);
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered) {
                impl->hovered = false;
                invalidate(Dirty::Layout);
            }
            return mouseEventChildren(event);
        default:
            return mouseEventChildren(event);
    }
}

}  // namespace Jetstream::Sakura
