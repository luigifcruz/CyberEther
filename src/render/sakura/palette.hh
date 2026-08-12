#ifndef JETSTREAM_RENDER_SAKURA_PALETTE_PRIVATE_HH
#define JETSTREAM_RENDER_SAKURA_PALETTE_PRIVATE_HH

#include "context.hh"

namespace Jetstream::Sakura {

ColorRGBA<F32> ResolveColor(const Context& ctx, const std::string& key, const ColorRGBA<F32>& fallback);
ColorRGBA<F32> ResolveColor(const Context& ctx, const std::string& key);

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_PALETTE_PRIVATE_HH
