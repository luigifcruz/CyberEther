#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_MAPS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_MAPS_HH

#include "../components/compact_button.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/render/components/map_options.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct MapsSettingsView {
    using Rendering = Render::Components::MapRenderingOptions;

    struct Config {
        bool highResolution = true;
        Rendering rendering;
        bool cacheAvailable = false;
        std::optional<U64> cacheBytes;
        bool measuring = false;
        bool clearing = false;
        Sakura::Callback<void(bool)> onHighResolutionChange;
        Sakura::Callback<void()> onClearCache;
        Sakura::Callback<void(const Rendering&)> onRenderingChange;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    struct Layer {
        const char* label;
        const char* description;
        bool Rendering::*field;
    };

    static constexpr std::array<Layer, 6> Layers = {{
        {"Roads and Railways", "Show streets, highways, and railway lines.", &Rendering::roads},
        {"Rivers", "Show rivers, canals, and streams.", &Rendering::rivers},
        {"Country and State Borders", "Show national, regional, and disputed boundaries.", &Rendering::borders},
        {"Place Labels", "Show geographic names, city markers, and airport labels.", &Rendering::places},
        {"Airports", "Show airport areas, runways, taxiways, and airport markers.", &Rendering::airports},
        {"Ocean Depth Shading", "Shade ocean areas by depth.", &Rendering::bathymetry},
    }};

    void renderingChanged();
    std::string cacheLabel() const;
    static std::string SizeLabel(U64 bytes);

    Config config;
    SettingsHeader header;
    Sakura::Toggle highResolutionToggle;
    Sakura::PillTabs buildingsTabs;
    std::array<Sakura::Toggle, Layers.size()> layerToggles;
    CompactButton clearButton;
    SectionGroup detailGroup;
    SectionGroup renderingGroup;
    SectionGroup storageGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_MAPS_HH
