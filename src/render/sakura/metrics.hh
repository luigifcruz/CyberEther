#ifndef JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH
#define JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH

#include "context.hh"

#include <jetstream/render/sakura/surface.hh>

#include <jetstream/render/sakura/typography.hh>

#include <algorithm>
#include <cmath>
#include <optional>

namespace Jetstream::Sakura {

constexpr F32 kFramePaddingX = 12.0f;
constexpr F32 kCellPaddingX = 6.0f;
constexpr F32 kCellPaddingY = 4.0f;
constexpr F32 kFramePaddingY = 5.0f;
constexpr F32 kFrameRounding = 8.0f;
constexpr F32 kPanelRounding = 12.0f;
constexpr F32 kPopupRounding = 18.0f;
constexpr F32 kWindowPadding = 12.0f;
constexpr F32 kItemSpacing = 8.0f;
constexpr F32 kDisabledAlpha = 0.6f;
constexpr F32 kShadowSoftness = 6.0f;
constexpr F32 kShadowDrop = 2.0f;
constexpr F32 kScrollbarThickness = 8.0f;
constexpr F32 kScrollbarMargin = 3.0f;
constexpr F32 kWheelStep = 48.0f;
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
constexpr F32 kNodeFont = Typography::FontSize * Typography::ChromeScale;
constexpr F32 kNodeFramePadX = 6.0f;
constexpr F32 kNodeFramePadY = 3.0f;
constexpr F32 kNodeUnitInset = 3.0f;

inline F32 Px(const Context& ctx, F32 logical) {
    return logical * ctx.pixelRatio;
}

inline F32 Hairline(const Context& ctx) {
    return std::max(1.0f, std::round(ctx.pixelRatio));
}

inline ColorRGBA<F32> WithAlpha(ColorRGBA<F32> color, F32 alpha) {
    color.a *= std::clamp(alpha, 0.0f, 1.0f);
    return color;
}

inline F32 FontPx(const Context& ctx) {
    return Px(ctx, Typography::FontSize);
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
