#include "metrics.hh"

#include "helpers.hh"

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

F32 ScalingFactor(const Context& ctx) {
    return RenderWindow(ctx).scalingFactor();
}

F32 Scale(const Context& ctx, const F32 value) {
    return value * ScalingFactor(ctx);
}

F32 Unscale(const Context& ctx, const F32 value) {
    return value / ScalingFactor(ctx);
}

Extent2D<F32> Scale(const Context& ctx, const Extent2D<F32> value) {
    return {value.x < 0.0f ? value.x : Scale(ctx, value.x),
            value.y < 0.0f ? value.y : Scale(ctx, value.y)};
}

Extent2D<F32> Unscale(const Context& ctx, const Extent2D<F32> value) {
    return {value.x < 0.0f ? value.x : Unscale(ctx, value.x),
            value.y < 0.0f ? value.y : Unscale(ctx, value.y)};
}

Extent2D<F32> FramebufferScale(const Context& ctx) {
    return RenderWindow(ctx).framebufferScale();
}

Extent2D<U64> FramebufferSize(const Context& ctx, const Extent2D<F32>& displaySize) {
    return RenderWindow(ctx).framebufferSize(displaySize);
}

Extent2D<U64> LogicalFramebufferSize(const Context& ctx, const Extent2D<F32>& logicalSize) {
    return FramebufferSize(ctx, Scale(ctx, logicalSize));
}

F32 SurfaceScale(const Context& ctx) {
    return ScalingFactor(ctx) * FramebufferScale(ctx).x * 0.5f;
}

std::optional<SurfaceResize> ResolveSurfaceResize(const Context& ctx, const Extent2D<F32>& logicalSize) {
    if (logicalSize.x <= 0.0f || logicalSize.y <= 0.0f) {
        return std::nullopt;
    }

    const SurfaceResize resize{
        .logicalSize = {static_cast<U64>(logicalSize.x), static_cast<U64>(logicalSize.y)},
        .framebufferSize = LogicalFramebufferSize(ctx, logicalSize),
        .scale = SurfaceScale(ctx),
    };

    if (resize.logicalSize.x == 0 || resize.logicalSize.y == 0 ||
        resize.framebufferSize.x == 0 || resize.framebufferSize.y == 0) {
        return std::nullopt;
    }

    return resize;
}

F32 FrameRate() {
    return ImGui::GetIO().Framerate;
}

}  // namespace Jetstream::Sakura
