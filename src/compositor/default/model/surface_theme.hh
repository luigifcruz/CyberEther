#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_SURFACE_THEME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_SURFACE_THEME_HH

#include "jetstream/module_surface.hh"
#include "jetstream/render/sakura/palette.hh"
#include "jetstream/surface.hh"

#include "render/sakura/themes.hh"

#include <memory>
#include <unordered_map>

namespace Jetstream {

struct SurfaceTrack {
    std::weak_ptr<Module::Surface> surface;
    SurfaceEvent event;
};

using SurfaceTracks = std::unordered_map<const Module::Surface*, SurfaceTrack>;

inline void RethemeSurfaces(SurfaceTracks& tracks, const Sakura::Palette& palette) {
    for (auto it = tracks.begin(); it != tracks.end();) {
        const auto surface = it->second.surface.lock();
        if (!surface) {
            it = tracks.erase(it);
            continue;
        }
        it->second.event.theme = Sakura::MakeSurfaceTheme(palette, it->second.event.placement);
        surface->pushSurfaceEvent(it->second.event);
        ++it;
    }
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_SURFACE_THEME_HH
