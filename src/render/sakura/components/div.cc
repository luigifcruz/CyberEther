#include <jetstream/render/sakura/components/div.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/polyline.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../metrics.hh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kWavePeriod = 56.0f;
constexpr F32 kWaveRow = 28.0f;
constexpr F32 kWaveAmplitude = 10.5f;
constexpr F32 kWaveThickness = 1.3f;
constexpr F32 kWaveAlpha = 0.25f;
constexpr U64 kWaveCapacity = 65536;
constexpr F32 kCornerChord = 1.0f - 1.0f / std::numbers::sqrt2_v<F32>;
constexpr F32 kFloatShadowSoftness = 18.0f;
constexpr F32 kFloatShadowDrop = 6.0f;
constexpr const char* kShadowColorKey = "tile_shadow";

}  // namespace

struct Div::Impl {
    Config config;
    Box shadow;
    Box frame;
    Card card;
    Polyline waves;
    bool hovered = false;

    static U64 waveCapacity(const Context& ctx) {
        const F32 row = std::max(Px(ctx, kWaveRow), 1.0f);
        const U64 rows = static_cast<U64>(std::ceil(static_cast<F32>(ctx.framebufferSize.y) / row)) + 2;
        return std::max(kWaveCapacity, std::bit_ceil(WaveRowCapacity(ctx, Px(ctx, kWavePeriod)) * rows));
    }

    std::vector<Polyline::Instance> waveRows(const Context& ctx,
                                             const Rect& area,
                                             const Rect& visible,
                                             ColorRGBA<F32> color,
                                             U64 capacity) const {
        std::vector<Polyline::Instance> rows;
        const F32 period = Px(ctx, kWavePeriod);
        const F32 row = Px(ctx, kWaveRow);
        const F32 amplitude = Px(ctx, kWaveAmplitude);
        const F32 thickness = Px(ctx, kWaveThickness);
        if (area.empty() || visible.empty() || period <= 0.0f || row <= 0.0f) {
            return rows;
        }
        const F32 reach = amplitude + thickness;
        U64 vertices = 0;
        for (F32 center = area.y + row * 0.5f; center - amplitude < area.bottom(); center += row) {
            if (center + reach < visible.y || center - reach > visible.bottom()) {
                continue;
            }
            auto points = WavePoints(area.x, center, area.width, period, amplitude, visible.x, visible.right());
            vertices += kWavePointVertices * points.size();
            if (vertices > capacity) {
                break;
            }
            rows.push_back({.points = std::move(points), .color = color, .thickness = thickness});
        }
        return rows;
    }

    bool interactive() const {
        return static_cast<bool>(config.onClick) || static_cast<bool>(config.onDoubleClick);
    }
};

Div::Div() {
    impl = std::make_unique<Impl>();
    addChild(impl->shadow);
    addChild(impl->frame);
    addChild(impl->card);
}

Div::~Div() = default;

void Div::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children = {&impl->shadow, &impl->frame, &impl->card, &impl->waves};
    impl->waves.setVisible(config.waves);
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
    const bool shadowed = config.shadow && !config.chrome;
    const F32 softness = Px(ctx, kFloatShadowSoftness);
    const F32 drop = Px(ctx, kFloatShadowDrop);
    impl->shadow.setVisible(shadowed);
    impl->shadow.update({
        .instances = {{
            .rect = Inflate({bounds.x, bounds.y + drop, bounds.width, bounds.height}, softness),
            .visible = shadowed && !bounds.empty(),
            .backgroundColor = ctx.color(kShadowColorKey),
        }},
        .clip = clip(),
        .cornerRadius = Px(ctx, config.rounding),
        .softness = softness,
    });
    layoutChild(ctx, impl->shadow, bounds);
    impl->frame.setVisible(!config.chrome);
    impl->frame.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color(key)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = config.border ? Px(ctx, config.borderWidth) : 0.0f,
        .borderColor = ctx.color(config.borderColorKey),
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->frame, bounds);
    impl->card.setVisible(config.chrome);
    impl->card.update({
        .rects = {bounds},
        .clip = clip(),
        .cornerRadius = Px(ctx, config.rounding),
        .softness = config.shadow ? softness : 0.0f,
        .drop = config.shadow ? drop : 0.0f,
        .borderWidth = config.border ? Px(ctx, config.borderWidth) : 0.0f,
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = key,
        .capacity = 1,
    });
    layoutChild(ctx, impl->card, bounds);
    if (config.waves) {
        const F32 corner = Px(ctx, config.rounding) * kCornerChord;
        const Rect area = Inset(bounds, std::max(corner, config.border ? Px(ctx, config.borderWidth) : 0.0f));
        const U64 capacity = Impl::waveCapacity(ctx);
        impl->waves.update({
            .instances = impl->waveRows(ctx, bounds, Intersect(bounds, clip()),
                                        WithAlpha(ctx.color(config.borderColorKey), kWaveAlpha), capacity),
            .clip = area,
            .fringe = Fringe(ctx),
            .capacity = capacity,
        });
        layoutChild(ctx, impl->waves, bounds);
    }
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

}  // namespace Jetstream::Sakura
