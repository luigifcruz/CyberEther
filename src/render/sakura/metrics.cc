#include "metrics.hh"

#include "helpers.hh"

#include <cmath>

namespace Jetstream::Sakura {

namespace {

const Render::Window& RenderWindow(const Context& ctx) {
    if (!ctx.render) {
        JST_FATAL("Sakura::Context is missing render window.");
        std::abort();
    }
    return *ctx.render;
}

}  // namespace

const Render::WindowMetrics& Metrics(const Context& ctx) {
    return RenderWindow(ctx).metrics();
}

F32 ScalingFactor(const Context& ctx) {
    return RenderWindow(ctx).scalingFactor();
}

F32 Scale(const Context& ctx, const F32 value) {
    return value * ScalingFactor(ctx);
}

Extent2D<F32> Scale(const Context& ctx, const Extent2D<F32> value) {
    return {value.x < 0.0f ? value.x : Scale(ctx, value.x),
            value.y < 0.0f ? value.y : Scale(ctx, value.y)};
}

Extent2D<F32> FramebufferScale(const Context& ctx) {
    return RenderWindow(ctx).framebufferScale();
}

Rect DisplayMetrics::RootRegion(const Render::WindowMetrics& metrics, F32 topOffset) {
    const F32 top = topOffset * metrics.scalingFactor;
    return {0.0f, std::trunc(top), std::trunc(metrics.displaySize.x),
            std::trunc(std::max(0.0f, metrics.displaySize.y - top))};
}

Rect DisplayMetrics::RootClip(const Render::WindowMetrics& metrics, const Rect& region) {
    const F32 left = std::max(0.0f, std::floor(0.5f + region.x));
    const F32 top = std::max(0.0f, std::floor(0.5f + region.y));
    const F32 right = std::min(metrics.displaySize.x, std::floor(region.right()));
    const F32 bottom = std::min(metrics.displaySize.y, std::floor(region.bottom()));
    return {left, top, std::max(0.0f, right - left), std::max(0.0f, bottom - top)};
}

Extent2D<U64> DisplayMetrics::CanvasFramebuffer(const Render::WindowMetrics& metrics, const Extent2D<F32>& logical) {
    constexpr F32 kSnapEpsilon = 1e-3f;
    const Extent2D<F32> display = {logical.x * metrics.scalingFactor, logical.y * metrics.scalingFactor};
    return {
        static_cast<U64>(std::floor(display.x * metrics.framebufferScale.x + kSnapEpsilon)),
        static_cast<U64>(std::floor(display.y * metrics.framebufferScale.y + kSnapEpsilon)),
    };
}

Extent2D<F32> DisplayMetrics::FramebufferToDisplay(const Render::WindowMetrics& metrics,
                                                   const Extent2D<U64>& framebuffer) {
    const auto& scale = metrics.framebufferScale;
    return {scale.x > 0.0f ? static_cast<F32>(framebuffer.x) / scale.x : 0.0f,
            scale.y > 0.0f ? static_cast<F32>(framebuffer.y) / scale.y : 0.0f};
}

Extent2D<F32> DisplayMetrics::LogicalSize(const Render::WindowMetrics& metrics, const Extent2D<F32>& display) {
    return {std::max(0.0f, display.x / metrics.scalingFactor), std::max(0.0f, display.y / metrics.scalingFactor)};
}

Extent2D<F32> DisplayMetrics::Snap(const Render::WindowMetrics& metrics, const Extent2D<F32>& display) {
    const auto& scale = metrics.framebufferScale;
    if (scale.x <= 0.0f || scale.y <= 0.0f) {
        return display;
    }
    return {std::round(display.x * scale.x) / scale.x, std::round(display.y * scale.y) / scale.y};
}

F32 DisplayMetrics::SurfaceScale(const Render::WindowMetrics& metrics) {
    return metrics.pixelRatio * kSurfaceScale;
}

F32 DisplayMetrics::PixelRatio(const SurfaceResize& resize) {
    return resize.scale > 0.0f ? resize.scale / kSurfaceScale : 1.0f;
}

F32 DisplayMetrics::LogicalHeight(const Render::WindowMetrics& metrics, U64 framebufferHeight) {
    return static_cast<F32>(framebufferHeight) /
           std::max(1e-3f, metrics.scalingFactor * metrics.framebufferScale.y);
}

std::optional<SurfaceResize> ResolveSurfaceResize(const Context& ctx, const Extent2D<F32>& logicalSize) {
    if (logicalSize.x <= 0.0f || logicalSize.y <= 0.0f) {
        return std::nullopt;
    }

    const SurfaceResize resize{
        .logicalSize = {static_cast<U64>(logicalSize.x), static_cast<U64>(logicalSize.y)},
        .framebufferSize = DisplayMetrics::CanvasFramebuffer(Metrics(ctx), logicalSize),
        .scale = DisplayMetrics::SurfaceScale(Metrics(ctx)),
    };

    if (resize.logicalSize.x == 0 || resize.logicalSize.y == 0 ||
        resize.framebufferSize.x == 0 || resize.framebufferSize.y == 0) {
        return std::nullopt;
    }

    return resize;
}

F32 LogicalViewportHeight(const Context& ctx) {
    return DisplayMetrics::LogicalHeight(Metrics(ctx), ctx.framebufferSize.y);
}

}  // namespace Jetstream::Sakura
