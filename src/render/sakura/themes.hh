#ifndef JETSTREAM_RENDER_SAKURA_THEMES_HH
#define JETSTREAM_RENDER_SAKURA_THEMES_HH

#include <jetstream/render/sakura/palette.hh>
#include <jetstream/surface.hh>

#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream::Sakura {

extern const std::unordered_map<std::string, Palette> Themes;

std::string ResolveThemeKey(const std::string& value);
std::vector<std::string> BuildThemeKeys();
SurfaceTheme MakeSurfaceTheme(const Palette& palette, SurfacePlacementType placement);

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_THEMES_HH
