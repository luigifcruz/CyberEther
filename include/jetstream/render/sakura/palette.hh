#ifndef JETSTREAM_RENDER_SAKURA_PALETTE_HH
#define JETSTREAM_RENDER_SAKURA_PALETTE_HH

#include <jetstream/types.hh>

#include <string>
#include <unordered_map>

namespace Jetstream::Sakura {

using Palette = std::unordered_map<std::string, ColorRGBA<F32>>;

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_PALETTE_HH
