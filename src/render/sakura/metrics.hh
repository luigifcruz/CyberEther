#ifndef JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH
#define JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH

#include "context.hh"

#include <jetstream/render/sakura/metrics.hh>
#include <jetstream/render/sakura/surface.hh>

#include <optional>

namespace Jetstream::Sakura {

F32 ScalingFactor(const Context& ctx);
F32 Unscale(const Context& ctx, F32 value);
Extent2D<F32> Unscale(const Context& ctx, Extent2D<F32> value);
Extent2D<F32> FramebufferScale(const Context& ctx);
Extent2D<U64> FramebufferSize(const Context& ctx, const Extent2D<F32>& displaySize);
Extent2D<U64> LogicalFramebufferSize(const Context& ctx, const Extent2D<F32>& logicalSize);
F32 SurfaceScale(const Context& ctx);
std::optional<SurfaceResize> ResolveSurfaceResize(const Context& ctx, const Extent2D<F32>& logicalSize);

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_METRICS_PRIVATE_HH
