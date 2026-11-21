#include <jetstream/render/sakura/components/outline.hh>

#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kOutlineText = 15.0f;
constexpr F32 kOutlineRowPadY = 5.0f;
constexpr F32 kOutlinePadLeft = 14.0f;
constexpr F32 kOutlineIndent = 12.0f;
constexpr F32 kOutlineLineWidth = 1.5f;
constexpr F64 kOutlineGlide = 0.14;

}  // namespace

struct Outline::Impl {
    Config config;
    Polyline track;
    Polyline marker;
    std::vector<std::unique_ptr<Text>> rows;
    std::vector<Rect> rects;
    std::vector<Extent2D<F32>> path;
    std::optional<U64> hovered;
    std::optional<U64> pressed;
    std::optional<U64> markerIndex;
    F32 markerTop = 0.0f;
    F32 markerHeight = 0.0f;
    F32 glideTop = 0.0f;
    F32 glideHeight = 0.0f;
    std::optional<F64> glideStart;

    void style() {
        for (U64 index = 0; index < config.entries.size(); ++index) {
            const bool lit = config.active == index || hovered == index;
            rows[index]->update({
                .str = config.entries[index].label,
                .font = Text::Font::Sans,
                .colorKey = lit ? "text_primary" : "text_secondary",
                .wrapped = true,
                .scale = kOutlineText / Typography::FontSize,
                .weight = config.active == index ? kLabelWeight : 0.0f,
            });
        }
    }

    std::vector<Extent2D<F32>> slice(F32 top, F32 bottom) const {
        std::vector<Extent2D<F32>> points;
        const auto at = [](const Extent2D<F32>& from, const Extent2D<F32>& to, F32 y) {
            const F32 span = to.y - from.y;
            const F32 t = span > 0.0f ? (y - from.y) / span : 0.0f;
            return Extent2D<F32>{from.x + (to.x - from.x) * t, y};
        };
        for (U64 index = 1; index < path.size(); ++index) {
            const auto& from = path[index - 1];
            const auto& to = path[index];
            if (to.y <= top || from.y >= bottom) {
                continue;
            }
            if (points.empty()) {
                points.push_back(from.y >= top ? from : at(from, to, top));
            }
            points.push_back(to.y <= bottom ? to : at(from, to, bottom));
        }
        return points;
    }

    std::optional<U64> rowAt(const Extent2D<F32>& point, const Rect& visible) const {
        if (!visible.contains(point.x, point.y)) {
            return std::nullopt;
        }
        for (U64 index = 0; index < rects.size(); ++index) {
            if (rects[index].contains(point.x, point.y)) {
                return index;
            }
        }
        return std::nullopt;
    }
};

Outline::Outline() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->track, &impl->marker});
}

Outline::~Outline() = default;

void Outline::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool resized = config.entries.size() != impl->config.entries.size();
    impl->config = std::move(config);
    while (impl->rows.size() < impl->config.entries.size()) {
        impl->rows.push_back(std::make_unique<Text>());
    }
    if (resized) {
        impl->hovered.reset();
        impl->pressed.reset();
        impl->rects.clear();
        std::vector<Component*> active = {&impl->track, &impl->marker};
        for (U64 index = 0; index < impl->config.entries.size(); ++index) {
            active.push_back(impl->rows[index].get());
        }
        mountChildren(active);
    }
    impl->style();
}

Component* Outline::row(U64 index) const {
    return index < impl->config.entries.size() ? impl->rows[index].get() : nullptr;
}

Extent2D<F32> Outline::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 padY = std::round(Px(ctx, kOutlineRowPadY));
    F32 width = 0.0f;
    F32 height = 0.0f;
    for (U64 index = 0; index < impl->config.entries.size(); ++index) {
        const F32 indent = kOutlineIndent * static_cast<F32>(impl->config.entries[index].level);
        const F32 inset = Px(ctx, kOutlinePadLeft + indent);
        const F32 inner = std::isfinite(available.x) ? std::max(0.0f, available.x - inset) : infinite;
        const auto text = measureChild(*impl->rows[index], ctx, {inner, infinite});
        width = std::max(width, text.x + inset);
        height += text.y + 2.0f * padY;
    }
    return {FillLength(available.x, std::ceil(width)), FitLength(std::ceil(height), available.y)};
}

void Outline::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 padY = std::round(Px(ctx, kOutlineRowPadY));

    s.rects.resize(s.config.entries.size());
    F32 cursor = bounds.y;
    for (U64 index = 0; index < s.config.entries.size(); ++index) {
        const F32 inset = Px(ctx, kOutlinePadLeft + kOutlineIndent * static_cast<F32>(s.config.entries[index].level));
        const F32 inner = std::max(0.0f, bounds.width - inset);
        const auto text = measureChild(*s.rows[index], ctx, {inner, infinite});
        const F32 height = text.y + 2.0f * padY;
        s.rects[index] = {bounds.x, cursor, bounds.width, height};
        layoutChild(ctx, *s.rows[index], {bounds.x + inset, std::round(cursor + padY), inner, text.y});
        cursor += height;
    }

    const F32 thickness = std::max(1.0f, std::round(Px(ctx, kOutlineLineWidth)));
    const F32 align = std::fmod(thickness, 2.0f) * 0.5f;
    const F32 slant = padY;
    s.path.clear();
    for (U64 index = 0; index < s.rects.size(); ++index) {
        const Rect& row = s.rects[index];
        const F32 x = std::floor(bounds.x + thickness +
                                 Px(ctx, kOutlineIndent * static_cast<F32>(s.config.entries[index].level))) + align;
        const bool joined = index > 0 && s.config.entries[index - 1].level != s.config.entries[index].level;
        const bool leaving = index + 1 < s.rects.size() &&
                             s.config.entries[index + 1].level != s.config.entries[index].level;
        if (index == 0) {
            s.path.push_back({x, row.y});
        } else if (joined) {
            s.path.push_back({x, row.y + slant});
        }
        if (leaving) {
            s.path.push_back({x, row.bottom() - slant});
        } else if (index + 1 == s.rects.size()) {
            s.path.push_back({x, row.bottom()});
        }
    }
    s.track.update({
        .instances = {{.points = s.path,
                       .color = ctx.color("border"),
                       .thickness = thickness,
                       .visible = s.path.size() > 1}},
        .fringe = Fringe(ctx),
    });
    layoutChild(ctx, s.track, bounds);

    const bool lit = s.config.active.has_value() && *s.config.active < s.rects.size();
    if (lit) {
        const Rect& target = s.rects[*s.config.active];
        const F32 targetTop = target.y - bounds.y;
        if (s.markerIndex != s.config.active) {
            if (s.markerIndex.has_value()) {
                s.glideTop = s.markerTop;
                s.glideHeight = s.markerHeight;
                s.glideStart = ctx.time;
            }
            s.markerIndex = s.config.active;
        }
        s.markerTop = targetTop;
        s.markerHeight = target.height;
        if (s.glideStart.has_value()) {
            const F32 t = static_cast<F32>(std::clamp((ctx.time - *s.glideStart) / kOutlineGlide, 0.0, 1.0));
            const F32 eased = Smoothstep(t);
            s.markerTop = s.glideTop + (targetTop - s.glideTop) * eased;
            s.markerHeight = s.glideHeight + (target.height - s.glideHeight) * eased;
            if (t < 1.0f) {
                requestFrame();
            } else {
                s.glideStart.reset();
            }
        }
    } else {
        s.markerIndex.reset();
        s.glideStart.reset();
    }
    s.marker.update({
        .instances = {{.points = lit ? s.slice(bounds.y + s.markerTop, bounds.y + s.markerTop + s.markerHeight)
                                     : std::vector<Extent2D<F32>>{},
                       .color = ctx.color("cyber_blue"),
                       .thickness = thickness,
                       .visible = lit}},
        .fringe = Fringe(ctx),
    });
    layoutChild(ctx, s.marker, bounds);
}

bool Outline::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    const auto row = s.rowAt(event.position, Intersect(frame(), clip()));
    switch (event.type) {
        case MouseEventType::Move:
            if (row != s.hovered) {
                s.hovered = row;
                s.style();
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (!row.has_value() || event.button != MouseButton::Left) {
                return false;
            }
            s.pressed = row;
            return true;
        case MouseEventType::Release:
            if (event.button != MouseButton::Left || !s.pressed.has_value()) {
                return false;
            }
            if (row == s.pressed && s.config.onSelect) {
                s.config.onSelect(*row);
            }
            s.pressed.reset();
            return true;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered.has_value() || s.pressed.has_value()) {
                s.hovered.reset();
                s.pressed.reset();
                s.style();
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Outline::cursorAt(const Extent2D<F32>& point) const {
    return impl->rowAt(point, Intersect(frame(), clip())).has_value() ? std::optional<SurfaceCursor>(SurfaceCursor::Hand) : std::nullopt;
}

}  // namespace Jetstream::Sakura
