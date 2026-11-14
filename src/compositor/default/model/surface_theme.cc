#include "surface_theme.hh"

namespace Jetstream {

void RethemeSurfaces(SurfaceTracks& tracks, const Sakura::Palette& palette) {
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
