#ifndef JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH
#define JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH

#include "context.hh"

#include <jetstream/render/sakura/components/dock.hh>
#include <jetstream/render/sakura/surface.hh>

#include <jetstream/render/sakura/typography.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <string_view>
#include <vector>

namespace Jetstream::Sakura {

constexpr F32 kFramePaddingX = 12.0f;
constexpr F32 kCellPaddingX = 6.0f;
constexpr F32 kCellPaddingY = 4.0f;
constexpr F32 kFramePaddingY = 5.0f;
constexpr F32 kFrameRounding = 8.0f;
constexpr F32 kPanelRounding = Dock::TileRounding;
constexpr F32 kCardRounding = 12.0f;
constexpr F32 kCardPadX = 18.0f;
constexpr F32 kLabelWeight = 0.15f;
constexpr F32 kPopupRounding = 18.0f;
constexpr F32 kWindowPadding = 12.0f;
constexpr F32 kItemSpacing = 8.0f;
constexpr F32 kTooltipPadding = 8.0f;
constexpr F32 kDisabledAlpha = 0.6f;
constexpr F32 kShadowSoftness = 6.0f;
constexpr F32 kShadowDrop = 2.0f;
constexpr F32 kScrollbarThickness = 8.0f;
constexpr F32 kScrollbarMargin = 3.0f;
constexpr F32 kWheelStep = 48.0f;
constexpr F32 kPointsPerWheelUnit = 10.0f;
constexpr F32 kInnerSpacing = 8.0f;
constexpr F32 kNaturalControlWidth = 200.0f;
constexpr F32 kScrollbarSize = 12.0f;
constexpr F32 kScrollbarGrabInset = 3.0f;
constexpr F32 kPillHeight = 34.0f;
constexpr F32 kPillIconGap = 8.0f;
constexpr F32 kPillShadowSoftness = 4.0f;
constexpr F32 kPillShadowDrop = 1.0f;
constexpr F32 kPillBorder = 1.5f;
constexpr const char* kPillFill = "pill_field_bg";
constexpr const char* kPillHover = "chrome_button_hover";
constexpr const char* kCardSurface = "welcome_card_bg";
constexpr F32 kNodeFont = Typography::FontSize * Typography::ChromeScale;
constexpr F32 kNodeFramePadX = 6.0f;
constexpr F32 kNodeFramePadY = 3.0f;
constexpr F32 kNodeUnitInset = 3.0f;
constexpr U64 kWaveSteps = 12;
constexpr U64 kWavePointVertices = 4;
constexpr U64 kWaveMaxPoints = 262144;
constexpr F32 kTwoPi = 2.0f * std::numbers::pi_v<F32>;

inline F32 Px(const Context& ctx, F32 logical) {
    return logical * ctx.pixelRatio;
}

inline F32 Hairline(const Context& ctx) {
    return std::max(1.0f, std::round(ctx.pixelRatio));
}

inline F32 WheelStep(const MouseEvent& event, F32 wheelStep, F32 pixelRatio) {
    return event.phase != ScrollPhase::None || event.precise ? kPointsPerWheelUnit * pixelRatio : wheelStep;
}

inline ColorRGBA<F32> WithAlpha(ColorRGBA<F32> color, F32 alpha) {
    color.a *= std::clamp(alpha, 0.0f, 1.0f);
    return color;
}

constexpr F32 kRimGain = 1.6f;
constexpr F32 kRimLift = 0.045f;
constexpr F32 kRimShade = 0.92f;
constexpr F32 kRimPivot = 0.8f;

inline ColorRGBA<F32> RimColor(ColorRGBA<F32> fill) {
    const F32 luma = 0.2126f * fill.r + 0.7152f * fill.g + 0.0722f * fill.b;
    const F32 gain = luma < kRimPivot ? kRimGain : kRimShade;
    const F32 lift = luma < kRimPivot ? kRimLift : 0.0f;
    return {std::min(fill.r * gain + lift, 1.0f), std::min(fill.g * gain + lift, 1.0f),
            std::min(fill.b * gain + lift, 1.0f), fill.a};
}

inline ColorRGBA<F32> BlendOver(ColorRGBA<F32> base, ColorRGBA<F32> over) {
    return {base.r + (over.r - base.r) * over.a, base.g + (over.g - base.g) * over.a,
            base.b + (over.b - base.b) * over.a, 1.0f};
}

inline ColorRGBA<F32> MixColor(ColorRGBA<F32> from, ColorRGBA<F32> to, F32 t) {
    return {from.r + (to.r - from.r) * t, from.g + (to.g - from.g) * t, from.b + (to.b - from.b) * t,
            from.a + (to.a - from.a) * t};
}

inline F32 Smoothstep(F32 t) {
    return t * t * (3.0f - 2.0f * t);
}

inline Context CardSurface(const Context& ctx) {
    Context inner = ctx;
    inner.backdropColorKey = kCardSurface;
    inner.backdropTopColorKey.clear();
    return inner;
}

inline F64 WaveSampleStep(F32 period) {
    return std::isfinite(period) && period > 0.0f ? static_cast<F64>(period) / kWaveSteps : 0.0;
}

inline std::vector<Extent2D<F32>> WavePoints(F32 left, F32 center, F32 width, F32 period, F32 amplitude, F32 from,
                                             F32 to) {
    std::vector<Extent2D<F32>> points;
    const F64 step = WaveSampleStep(period);
    if (!std::isfinite(left) || !std::isfinite(center) || !std::isfinite(width) || !std::isfinite(amplitude) ||
        !std::isfinite(from) || !std::isfinite(to) || width <= 0.0f || step == 0.0 || to < from) {
        return points;
    }
    const F64 begin = std::max(static_cast<F64>(from), static_cast<F64>(left));
    const F64 end = std::min(static_cast<F64>(to), static_cast<F64>(left) + width);
    if (end < begin) {
        return points;
    }
    const F64 first = std::floor((begin - left) / step);
    const F64 last = std::ceil((end - left) / step);
    constexpr U64 kMaxExactIndex = (U64{1} << std::numeric_limits<F64>::digits) - 1;
    constexpr F64 kMaxCoordinate = std::numeric_limits<F32>::max();
    if (last > static_cast<F64>(kMaxExactIndex) || last - first + 1.0 > kWaveMaxPoints ||
        std::abs(static_cast<F64>(left) + last * step) > kMaxCoordinate ||
        std::abs(static_cast<F64>(center)) + std::abs(static_cast<F64>(amplitude)) > kMaxCoordinate) {
        return points;
    }
    const U64 count = static_cast<U64>(last - first) + 1;
    const U64 start = static_cast<U64>(first);
    points.reserve(count);
    for (U64 offset = 0; offset < count; ++offset) {
        const U64 index = start + offset;
        const F64 x = static_cast<F64>(left) + static_cast<F64>(index) * step;
        const F64 phase = 2.0 * std::numbers::pi_v<F64> * static_cast<F64>(index % kWaveSteps) / kWaveSteps;
        const F64 y = static_cast<F64>(center) - static_cast<F64>(amplitude) * std::sin(phase);
        points.push_back({static_cast<F32>(x), static_cast<F32>(y)});
    }
    return points;
}

inline U64 WaveRowCapacity(const Context& ctx, F32 period) {
    const F64 step = WaveSampleStep(period);
    if (step == 0.0 || ctx.framebufferSize.x == 0) {
        return 0;
    }
    const F64 count = std::ceil(static_cast<F64>(ctx.framebufferSize.x) / step) + 3.0;
    return kWavePointVertices * static_cast<U64>(std::min(count, static_cast<F64>(kWaveMaxPoints)));
}

inline F32 FontPx(const Context& ctx) {
    return Px(ctx, Typography::FontSize);
}

inline bool IsMonospaceFont(std::string_view fontName) {
    return fontName == Typography::MonoFont || fontName == Typography::MonoBoldFont;
}

inline F32 NodeFontOf(const Context& ctx) {
    return ctx.fontSize > 0.0f ? ctx.fontSize : Typography::FontSize;
}

inline F32 NodeFrameHeight(const Context& ctx) {
    return NodeFontOf(ctx) + 2.0f * kNodeFramePadY;
}

inline F32 FrameHeightPx(const Context& ctx) {
    return Px(ctx, Typography::FontSize + 2.0f * kFramePaddingY);
}

inline F32 FillLength(F32 available, F32 natural = 0.0f) {
    return std::isfinite(available) ? available : natural;
}

inline F32 FitLength(F32 length, F32 available) {
    return std::isfinite(available) ? std::min(length, std::max(0.0f, available)) : length;
}

inline Extent2D<F32> FitExtent(Extent2D<F32> size, Extent2D<F32> available) {
    return {FitLength(size.x, available.x), FitLength(size.y, available.y)};
}

inline F32 ResolveLength(const Context& ctx, F32 size, F32 natural, F32 available) {
    const F32 length = size > 0.0f ? Px(ctx, size) : size < 0.0f ? FillLength(available, natural) : natural;
    return FitLength(length, available);
}

struct DisplayMetrics {
    static constexpr F32 kSurfaceScale = 0.5f;

    static Rect RootRegion(const Render::WindowMetrics& metrics, F32 topOffset);
    static Rect RootClip(const Render::WindowMetrics& metrics, const Rect& region);
    static Extent2D<U64> CanvasFramebuffer(const Render::WindowMetrics& metrics, const Extent2D<F32>& logical);
    static Extent2D<F32> FramebufferToDisplay(const Render::WindowMetrics& metrics, const Extent2D<U64>& framebuffer);
    static Extent2D<F32> LogicalSize(const Render::WindowMetrics& metrics, const Extent2D<F32>& display);
    static Extent2D<F32> Snap(const Render::WindowMetrics& metrics, const Extent2D<F32>& display);
    static F32 SurfaceScale(const Render::WindowMetrics& metrics);
    static F32 PixelRatio(const SurfaceResize& resize);
    static F32 LogicalHeight(const Render::WindowMetrics& metrics, U64 framebufferHeight);
};

const Render::WindowMetrics& Metrics(const Context& ctx);
F32 ScalingFactor(const Context& ctx);
F32 Scale(const Context& ctx, F32 value);
Extent2D<F32> Scale(const Context& ctx, Extent2D<F32> value);
Extent2D<F32> FramebufferScale(const Context& ctx);
std::optional<SurfaceResize> ResolveSurfaceResize(const Context& ctx, const Extent2D<F32>& logicalSize);
F32 LogicalViewportHeight(const Context& ctx);

inline F32 Fringe(const Context& ctx) {
    return ctx.render ? std::max(1e-3f, FramebufferScale(ctx).x) : 1.0f;
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH
