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

Extent2D<F32> SnapToFramebuffer(const Context& ctx, const Extent2D<F32>& displayPosition) {
    const auto scale = FramebufferScale(ctx);
    if (scale.x <= 0.0f || scale.y <= 0.0f) {
        return displayPosition;
    }
    return {std::round(displayPosition.x * scale.x) / scale.x,
            std::round(displayPosition.y * scale.y) / scale.y};
}

Extent2D<F32> FramebufferToDisplay(const Context& ctx, const Extent2D<U64>& framebufferSize) {
    const auto scale = FramebufferScale(ctx);
    return {scale.x > 0.0f ? static_cast<F32>(framebufferSize.x) / scale.x : 0.0f,
            scale.y > 0.0f ? static_cast<F32>(framebufferSize.y) / scale.y : 0.0f};
}

std::optional<SurfaceResize> ResolveSurfaceResize(const Context& ctx, const Extent2D<F32>& logicalSize) {
    if (logicalSize.x <= 0.0f || logicalSize.y <= 0.0f) {
        return std::nullopt;
    }

    constexpr F32 kSnapEpsilon = 1e-3f;
    const auto framebufferScale = FramebufferScale(ctx);
    const auto displaySize = Scale(ctx, logicalSize);
    const SurfaceResize resize{
        .logicalSize = {static_cast<U64>(logicalSize.x), static_cast<U64>(logicalSize.y)},
        .framebufferSize = {
            static_cast<U64>(std::floor(displaySize.x * framebufferScale.x + kSnapEpsilon)),
            static_cast<U64>(std::floor(displaySize.y * framebufferScale.y + kSnapEpsilon)),
        },
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
