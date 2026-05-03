#pragma once

#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <optional>

namespace Jetstream::Sakura {

struct SurfaceResize {
    Extent2D<U64> logicalSize = {0, 0};
    Extent2D<U64> framebufferSize = {0, 0};
    F32 scale = 1.0f;
};

struct SurfaceSize {
    Extent2D<F32> availableLogicalSize = {0.0f, 0.0f};
    Extent2D<F32> resolvedLogicalSize = {0.0f, 0.0f};
    Extent2D<U64> logicalSize = {0, 0};
    Extent2D<U64> framebufferSize = {0, 0};
    F32 scale = 1.0f;
};

std::optional<SurfaceResize> ResolveSurfaceResize(const Context& ctx, const Extent2D<F32>& logicalSize);

}  // namespace Jetstream::Sakura
