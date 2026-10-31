#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_MAPS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_MAPS_HH

#include "components/settings_group.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"
#include "jetstream/render/components/map_options.hh"

#include <array>
#include <functional>
#include <optional>
#include <string>

namespace Jetstream {

struct MapsSettingsPanel {
    using Rendering = Render::Components::MapRenderingOptions;

    struct Config {
        bool highResolution = true;
        Rendering rendering;
        bool cacheAvailable = false;
        std::optional<U64> cacheBytes;
        bool measuring = false;
        bool clearing = false;
        std::function<void(bool)> onHighResolutionChange;
        std::function<void()> onClearCache;
        std::function<void(const Rendering&)> onRenderingChange;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "MapsTitle",
            .str = "Maps",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .id = "MapsDescription",
            .str = "Configure map detail and downloaded maps.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "MapsHeaderDivider",
        });

        detailSection.update({
            .id = "MapsDetailSection",
            .title = "Detail",
            .description = "The world map is built into CyberEther and always works, even offline. "
                           "High resolution maps add buildings, streets, rivers, airports, and neighborhood names "
                           "when you zoom in.",
        });

        storageSection.update({
            .id = "MapsStorageSection",
            .title = "Storage",
            .description = "Places you have looked at are saved on this computer so they load faster next time.",
        });

        highResolutionField.update({
            .id = "MapsHighResolutionField",
            .label = "High Resolution Maps",
            .description = "Downloads detailed maps from OpenFreeMap as you zoom in. "
                           "Turn this off to use only the built-in world map.",
            .divider = false,
        });

        highResolutionCombo.update({
            .id = "##app-settings-maps-high-resolution",
            .options = {
                "Enabled",
                "Disabled",
            },
            .value = this->config.highResolution ? "Enabled" : "Disabled",
            .onChange = [this](const std::string& label) {
                if (this->config.onHighResolutionChange) {
                    this->config.onHighResolutionChange(label == "Enabled");
                }
            },
        });

        renderingSection.update({
            .id = "MapsRenderingSection",
            .title = "Rendering",
            .description = "Choose which map details to show. Changes apply to all maps immediately. "
                           "Buildings and roads require high resolution maps.",
        });
        buildingsField.update({
            .id = "MapsBuildingsField",
            .label = "Buildings",
            .description = "Show building footprints and optional outlines. Turning buildings off reduces "
                           "memory use in dense cities.",
        });
        buildingsCombo.update({
            .id = "##app-settings-maps-buildings",
            .options = {"Off", "Fills only", "Fills and outlines"},
            .value = !this->config.rendering.buildings ? "Off" :
                     this->config.rendering.buildingOutlines ? "Fills and outlines" : "Fills only",
            .disabled = !this->config.highResolution,
            .onChange = [this](const std::string& value) {
                this->config.rendering.buildings = value != "Off";
                this->config.rendering.buildingOutlines = value == "Fills and outlines";
                renderingChanged();
            },
        });
        for (U64 i = 0; i < Layers.size(); ++i) {
            const auto& layer = Layers[i];
            layerFields[i].update({
                .id = std::string("MapsRenderingField") + layer.id,
                .label = layer.label,
                .description = layer.description,
                .divider = i + 1 < Layers.size(),
            });
            layerCombos[i].update({
                .id = std::string("##app-settings-maps-") + layer.id,
                .options = {"Enabled", "Disabled"},
                .value = this->config.rendering.*layer.field ? "Enabled" : "Disabled",
                .disabled = layer.field == &Rendering::roads && !this->config.highResolution,
                .onChange = [this, field = layer.field](const std::string& value) {
                    this->config.rendering.*field = value == "Enabled";
                    renderingChanged();
                },
            });
        }

        cacheField.update({
            .id = "MapsCacheField",
            .label = "Downloaded Maps",
            .description = "Clearing is safe. Maps download again the next time you view those places.",
            .divider = false,
        });

        cacheSize.update({
            .id = "MapsCacheSize",
            .str = cacheLabel(),
            .tone = Sakura::Text::Tone::Secondary,
        });

        cacheSpacing.update({
            .id = "MapsCacheSpacing",
        });

        clearButton.update({
            .id = "MapsClearButton",
            .str = this->config.clearing ? "Clearing..." : "Clear Downloaded Maps",
            .size = {-1.0f, 0.0f},
            .variant = Sakura::Button::Variant::Destructive,
            .disabled = this->config.clearing || this->config.measuring ||
                        !this->config.cacheBytes.has_value() || this->config.cacheBytes.value() == 0,
            .onClick = [this]() {
                if (this->config.onClearCache) {
                    this->config.onClearCache();
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        detailSection.render(ctx, [&](const Sakura::Context& ctx) {
            highResolutionField.render(ctx, [&](const Sakura::Context& ctx) {
                highResolutionCombo.render(ctx);
            });
        });

        renderingSection.render(ctx, [&](const Sakura::Context& ctx) {
            buildingsField.render(ctx, [&](const Sakura::Context& ctx) {
                buildingsCombo.render(ctx);
            });
            for (U64 i = 0; i < Layers.size(); ++i) {
                layerFields[i].render(ctx, [&](const Sakura::Context& ctx) {
                    layerCombos[i].render(ctx);
                });
            }
        });

        storageSection.render(ctx, [&](const Sakura::Context& ctx) {
            cacheField.render(ctx, [&](const Sakura::Context& ctx) {
                cacheSize.render(ctx);
                cacheSpacing.render(ctx);
                clearButton.render(ctx);
            });
        });
    }

 private:
    struct Layer {
        const char* id;
        const char* label;
        const char* description;
        bool Rendering::*field;
    };
    static constexpr std::array<Layer, 6> Layers = {{
        {"roads", "Roads and Railways", "Show streets, highways, and railway lines.", &Rendering::roads},
        {"rivers", "Rivers", "Show rivers, canals, and streams.", &Rendering::rivers},
        {"borders", "Country and State Borders", "Show national, regional, and disputed boundaries.", &Rendering::borders},
        {"places", "Place Labels", "Show geographic names, city markers, and airport labels.", &Rendering::places},
        {"airports", "Airports", "Show airport areas, runways, taxiways, and airport markers.", &Rendering::airports},
        {"bathymetry", "Ocean Depth Shading", "Shade ocean areas by depth.", &Rendering::bathymetry},
    }};

    void renderingChanged() {
        if (config.onRenderingChange) config.onRenderingChange(config.rendering);
    }

    std::string cacheLabel() const {
        if (config.clearing) {
            return "Clearing downloaded maps...";
        }
        if (config.measuring) {
            return "Calculating size...";
        }
        if (!config.cacheAvailable || !config.cacheBytes.has_value()) {
            return "Saved maps are not available on this platform.";
        }
        const U64 bytes = config.cacheBytes.value();
        if (bytes == 0) {
            return "No maps downloaded yet.";
        }
        return jst::fmt::format("Using {} of storage.", sizeLabel(bytes));
    }

    static std::string sizeLabel(U64 bytes) {
        const F64 value = static_cast<F64>(bytes);
        if (value >= 1e9) return jst::fmt::format("{:.1f} GB", value / 1e9);
        if (value >= 1e6) return jst::fmt::format("{:.1f} MB", value / 1e6);
        if (value >= 1e3) return jst::fmt::format("{:.0f} KB", value / 1e3);
        return jst::fmt::format("{} bytes", bytes);
    }

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    SettingsGroup detailSection;
    SettingsGroup renderingSection;
    SettingsGroup storageSection;
    Sakura::SettingField highResolutionField;
    Sakura::SettingField cacheField;
    Sakura::Combo highResolutionCombo;
    Sakura::SettingField buildingsField;
    Sakura::Combo buildingsCombo;
    std::array<Sakura::SettingField, Layers.size()> layerFields;
    std::array<Sakura::Combo, Layers.size()> layerCombos;
    Sakura::Text cacheSize;
    Sakura::Spacing cacheSpacing;
    Sakura::Button clearButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_MAPS_HH
