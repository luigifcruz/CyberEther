#ifndef JETSTREAM_RENDER_SAKURA_METRICS_HH
#define JETSTREAM_RENDER_SAKURA_METRICS_HH

#include <jetstream/types.hh>

namespace Jetstream::Sakura {

struct Context;

F32 Scale(const Context& ctx, F32 value);
Extent2D<F32> Scale(const Context& ctx, Extent2D<F32> value);
F32 FrameRate();

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_METRICS_HH
