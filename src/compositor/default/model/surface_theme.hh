#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_SURFACE_THEME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_SURFACE_THEME_HH

#include "jetstream/module_surface.hh"
#include "jetstream/render/sakura/palette.hh"
#include "jetstream/surface.hh"

#include <memory>
#include <unordered_map>

namespace Jetstream {

struct SurfaceTrack {
    std::weak_ptr<Module::Surface> surface;
    SurfaceEvent event;
};

using SurfaceTracks = std::unordered_map<const Module::Surface*, SurfaceTrack>;

inline SurfaceTheme MakeSurfaceTheme(const Sakura::Palette& palette, SurfacePlacementType placement) {
    SurfaceTheme theme;
    const auto color = [&palette](const char* key, const ColorRGBA<F32>& fallback) {
        const auto it = palette.find(key);
        return it == palette.end() ? fallback : it->second;
    };
    theme.background = color(placement == SurfacePlacementType::Attached ? "node_background" : "tile_bg",
                             theme.background);
    theme.background.a = 1.0f;
    theme.foreground = color("text_primary", theme.foreground);
    theme.secondary = color("text_secondary", theme.secondary);
    theme.grid = color("plot_grid", theme.grid);
    theme.majorGrid = color("plot_grid_major", theme.majorGrid);
    theme.accent = color("plot_accent", theme.accent);
    theme.halo = color("plot_halo", theme.halo);
    theme.pill = color("pill_field_bg", theme.pill);
    return theme;
}

inline void RethemeSurfaces(SurfaceTracks& tracks, const Sakura::Palette& palette) {
    for (auto it = tracks.begin(); it != tracks.end();) {
        const auto surface = it->second.surface.lock();
        if (!surface) {
            it = tracks.erase(it);
            continue;
        }
        it->second.event.theme = MakeSurfaceTheme(palette, it->second.event.placement);
        surface->pushSurfaceEvent(it->second.event);
        ++it;
    }
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_SURFACE_THEME_HH
