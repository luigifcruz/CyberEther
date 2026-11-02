#include <jetstream/render/sakura/components/pill_tabs.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../detail/text_metrics.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

struct PillMetrics {
    F32 height;
    F32 padX;
    F32 rounding;
};

constexpr PillMetrics kRegular = {40.0f, 14.0f, 10.0f};
constexpr PillMetrics kCompact = {32.0f, 12.0f, 8.0f};
constexpr PillMetrics kTrack = {kPillHeight, 14.0f, kPillHeight * 0.5f};
constexpr F32 kTrackText = 14.0f;
constexpr F32 kTrackIcon = 13.0f;
constexpr F32 kTrackInset = 3.0f;
constexpr F32 kTrackSpacing = 4.0f;
constexpr F32 kPillTabText = 18.0f;
constexpr F32 kCompactIdleOpacity = 0.65f;
constexpr F32 kPillTabIconGap = 10.0f;
constexpr F32 kPillTabSpacing = 6.0f;
constexpr F32 kPillTabIcon = 17.0f;

}  // namespace

struct PillTabs::Impl {
    struct Entry {
        Text icon;
        Text label;
        std::string source;
        std::string fitted;
    };

    Config config;
    Card track;
    Box selection;
    Box hover;
    TextMetrics textMetrics;
    std::vector<std::unique_ptr<Entry>> entries;
    std::vector<Component*> mounted;
    std::vector<Rect> rects;
    std::optional<U64> hovered;
    std::optional<U64> pressed;

    const PillMetrics& metrics() const {
        return config.track ? kTrack : config.compact ? kCompact : kRegular;
    }

    F32 textSize() const {
        return config.track ? kTrackText : kPillTabText;
    }

    F32 iconSize() const {
        return config.track ? kTrackIcon : kPillTabIcon;
    }

    F32 spacing() const {
        return config.track ? kTrackSpacing : kPillTabSpacing;
    }

    F32 iconGap() const {
        return config.track ? kPillIconGap : kPillTabIconGap;
    }

    F32 inset() const {
        return config.track ? kTrackInset : 0.0f;
    }

    F32 labelWidth(const Context& ctx, U64 index) {
        return textMetrics.measure(Typography::DisplayFont, config.items[index].label, Px(ctx, textSize()));
    }

    void fitLabel(const Context& ctx, U64 index, F32 room) {
        auto& entry = *entries[index];
        const auto& label = config.items[index].label;
        const F32 size = Px(ctx, textSize());
        const std::string fitted =
            labelWidth(ctx, index) <= std::ceil(room) ? label : textMetrics.ellipsize(Typography::DisplayFont, label, size, room);
        if (fitted == entry.fitted) {
            return;
        }
        entry.fitted = fitted;
        auto next = entry.label.config();
        next.str = fitted;
        entry.label.update(std::move(next));
    }

    std::optional<U64> itemAt(const Extent2D<F32>& point) const {
        for (U64 index = 0; index < rects.size(); ++index) {
            if (rects[index].contains(point.x, point.y)) {
                return index;
            }
        }
        return std::nullopt;
    }

    F32 itemWidth(PillTabs& self, const Context& ctx, U64 index) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        auto& entry = *entries[index];
        const F32 icon = config.items[index].icon.empty() ? 0.0f
                                                          : self.measureChild(entry.icon, ctx, {infinite, infinite}).x +
                                                                Px(ctx, iconGap());
        return std::ceil(2.0f * Px(ctx, metrics().padX) + icon + labelWidth(ctx, index));
    }
};

PillTabs::PillTabs() {
    impl = std::make_unique<Impl>();
    setClipsChildren(false);
}

PillTabs::~PillTabs() = default;

void PillTabs::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& items = impl->config.items;
    while (impl->entries.size() < items.size()) {
        impl->entries.push_back(std::make_unique<Impl::Entry>());
    }
    if (impl->hovered && *impl->hovered >= items.size()) {
        impl->hovered.reset();
    }
    if (impl->pressed && *impl->pressed >= items.size()) {
        impl->pressed.reset();
    }

    std::vector<Component*> children;
    if (impl->config.track) {
        children.push_back(&impl->track);
    }
    children.push_back(&impl->hover);
    children.push_back(&impl->selection);
    for (U64 index = 0; index < items.size(); ++index) {
        auto& entry = *impl->entries[index];
        const bool selected = index == impl->config.selected;
        entry.icon.update({
            .str = items[index].icon,
            .colorKey = selected ? "cyber_blue" : "text_secondary",
            .scale = impl->iconSize() / Typography::FontSize,
        });
        const bool compact = impl->config.compact;
        if (entry.source != items[index].label) {
            entry.source = items[index].label;
            entry.fitted = items[index].label;
        }
        entry.label.update({
            .str = entry.fitted,
            .font = Text::Font::Display,
            .colorKey = "text_primary",
            .scale = impl->textSize() / Typography::FontSize,
            .opacity = compact && !selected ? kCompactIdleOpacity : 1.0f,
        });
        if (!items[index].icon.empty()) {
            children.push_back(&entry.icon);
        }
        children.push_back(&entry.label);
    }
    const bool owned = std::all_of(children.begin(), children.end(), [this](const Component* child) {
        return hasChild(*child);
    });
    if (children != impl->mounted || !owned) {
        mountChildren(children);
        impl->mounted = std::move(children);
    }
}

Extent2D<F32> PillTabs::measure(const Context& ctx, Extent2D<F32> available) {
    impl->textMetrics.setWindow(ctx.render);
    const auto& items = impl->config.items;
    F32 width = 0.0f;
    for (U64 index = 0; index < items.size(); ++index) {
        width += impl->itemWidth(*this, ctx, index);
    }
    if (items.size() > 1) {
        width += Px(ctx, impl->spacing()) * static_cast<F32>(items.size() - 1);
    }
    width += 2.0f * Px(ctx, impl->inset());
    return {FitLength(width, available.x), FitLength(Px(ctx, impl->metrics().height), available.y)};
}

void PillTabs::layout(const Context& ctx) {
    auto& s = *impl;
    s.textMetrics.setWindow(ctx.render);
    const Rect outer = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    if (s.config.track) {
        s.track.update({
            .rects = {outer},
            .clip = clip(),
            .cornerRadius = outer.height * 0.5f,
            .softness = Px(ctx, kPillShadowSoftness),
            .drop = Px(ctx, kPillShadowDrop),
            .borderWidth = Px(ctx, kPillBorder),
            .lineWidth = Px(ctx, 1.0f),
            .fillKey = kPillFill,
        });
        layoutChild(ctx, s.track, outer);
    }
    const F32 inset = Px(ctx, s.inset());
    const Rect bounds = {outer.x + inset, outer.y + inset, std::max(0.0f, outer.width - 2.0f * inset),
                         std::max(0.0f, outer.height - 2.0f * inset)};
    const F32 padX = Px(ctx, s.metrics().padX);
    const F32 rounding = s.config.track ? bounds.height * 0.5f : Px(ctx, s.metrics().rounding);
    s.rects.clear();
    const U64 count = s.config.items.size();
    const F32 spacing = Px(ctx, s.spacing());
    std::vector<F32> widths(count, 0.0f);
    F32 natural = 0.0f;
    for (U64 index = 0; index < count; ++index) {
        widths[index] = s.itemWidth(*this, ctx, index);
        natural += widths[index];
    }
    const F32 room = std::max(0.0f, bounds.width - spacing * static_cast<F32>(count > 1 ? count - 1 : 0));
    const F32 fit = natural > room && natural > 0.0f ? room / natural : 1.0f;
    F32 x = bounds.x;
    for (U64 index = 0; index < count; ++index) {
        auto& entry = *s.entries[index];
        const Rect rect = {x, bounds.y, std::floor(widths[index] * fit), bounds.height};
        s.rects.push_back(rect);
        F32 cursor = rect.x + padX;
        if (!s.config.items[index].icon.empty()) {
            const auto icon = measureChild(entry.icon, ctx, {infinite, infinite});
            layoutChild(ctx, entry.icon, {cursor, std::round(rect.y + (rect.height - icon.y) * 0.5f), icon.x, icon.y},
                        rect);
            cursor += icon.x + Px(ctx, s.iconGap());
        }
        const F32 labelRoom = std::max(0.0f, rect.right() - padX - cursor);
        s.fitLabel(ctx, index, labelRoom);
        const auto label = measureChild(entry.label, ctx, {infinite, infinite});
        layoutChild(ctx, entry.label,
                    {cursor, std::round(rect.y + (rect.height - label.y) * 0.5f), std::min(label.x, labelRoom), label.y},
                    rect);
        x = rect.right() + spacing;
    }

    const bool selected = s.config.selected < s.rects.size();
    s.selection.update({
        .instances = {{.rect = selected ? s.rects[s.config.selected] : Rect{},
                       .visible = selected,
                       .backgroundColor = ctx.color("pill_selected_bg")}},
        .clip = bounds,
        .cornerRadius = rounding,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("setting_card_border"),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.selection, bounds);
    const bool lit = s.hovered && *s.hovered < s.rects.size() && *s.hovered != s.config.selected;
    s.hover.update({
        .instances = {{.rect = lit ? s.rects[*s.hovered] : Rect{},
                       .visible = lit,
                       .backgroundColor = ctx.color("pill_hover_bg")}},
        .clip = bounds,
        .cornerRadius = rounding,
        .capacity = 1,
    });
    layoutChild(ctx, s.hover, bounds);
}

bool PillTabs::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    switch (event.type) {
        case MouseEventType::Move: {
            const auto item = s.itemAt(event.position);
            if (item != s.hovered) {
                s.hovered = item;
                invalidate(Dirty::Layout);
            }
            return false;
        }
        case MouseEventType::Click: {
            const auto item = s.itemAt(event.position);
            if (!item || event.button != MouseButton::Left) {
                return false;
            }
            s.pressed = item;
            return true;
        }
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left || !s.pressed) {
                return false;
            }
            const auto pressed = *s.pressed;
            s.pressed.reset();
            if (s.itemAt(event.position) == pressed && s.config.onSelect) {
                s.config.onSelect(pressed);
            }
            return true;
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered || s.pressed) {
                s.hovered.reset();
                s.pressed.reset();
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> PillTabs::cursorAt(const Extent2D<F32>& point) const {
    if (impl->itemAt(point)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
