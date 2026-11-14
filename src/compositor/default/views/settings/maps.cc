#include "maps.hh"

namespace Jetstream {

void MapsSettingsView::update(Config config) {
    this->config = std::move(config);

    header.update({
        .title = "Maps",
        .description = "Configure map detail and downloaded maps.",
    });

    highResolutionToggle.update({
        .value = this->config.highResolution,
        .onChange = [this](bool value) {
            if (this->config.onHighResolutionChange) {
                this->config.onHighResolutionChange(value);
            }
        },
    });

    detailGroup.update({
        .title = "Detail",
        .description = "The world map is built into CyberEther and always works, even offline. "
                       "High resolution maps add buildings, streets, rivers, airports, and neighborhood "
                       "names when you zoom in.",
        .rows = {{
            .label = "High Resolution Maps",
            .description = "Downloads detailed maps from OpenFreeMap as you zoom in. "
                           "Turn this off to use only the built-in world map.",
            .control = &highResolutionToggle,
        }},
    });

    const auto& rendering = this->config.rendering;
    buildingsTabs.update({
        .items = {{.label = "Off"}, {.label = "Fills"}, {.label = "Outlines"}},
        .selected = !rendering.buildings ? 0u : rendering.buildingOutlines ? 2u : 1u,
        .compact = true,
        .track = true,
        .onSelect = [this](U64 index) {
            this->config.rendering.buildings = index > 0;
            this->config.rendering.buildingOutlines = index == 2;
            renderingChanged();
        },
    });

    std::vector<Sakura::SettingCard::Row> rows = {{
        .label = "Buildings",
        .description = "Show building footprints with fills only, or with outlines. "
                       "Turning buildings off reduces memory use in dense cities.",
        .control = &buildingsTabs,
    }};
    for (U64 index = 0; index < Layers.size(); ++index) {
        const auto& layer = Layers[index];
        layerToggles[index].update({
            .value = rendering.*layer.field,
            .disabled = layer.field == &Rendering::roads && !this->config.highResolution,
            .onChange = [this, field = layer.field](bool value) {
                this->config.rendering.*field = value;
                renderingChanged();
            },
        });
        rows.push_back({
            .label = layer.label,
            .description = layer.description,
            .control = &layerToggles[index],
        });
    }
    renderingGroup.update({
        .title = "Rendering",
        .description = "Choose which map details to show. Changes apply to all maps immediately. "
                       "Buildings and roads require high resolution maps.",
        .rows = std::move(rows),
    });

    clearButton.update({
        .str = this->config.clearing ? "Clearing..." : "Clear",
        .icon = ICON_LC_TRASH_2,
        .tone = CompactButton::Tone::Danger,
        .disabled = this->config.clearing || this->config.measuring ||
                    !this->config.cacheBytes.has_value() || this->config.cacheBytes.value() == 0,
        .onClick = [this]() {
            if (this->config.onClearCache) {
                this->config.onClearCache();
            }
        },
    });

    storageGroup.update({
        .title = "Storage",
        .description = "Places you have looked at are saved on this computer so they load faster next time.",
        .rows = {{
            .label = "Downloaded Maps",
            .description = cacheLabel() + " Clearing is safe, maps download again the next time you view "
                                          "those places.",
            .control = &clearButton,
        }},
    });

    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &header},
            {.child = &detailGroup},
            {.child = &renderingGroup},
            {.child = &storageGroup},
        },
    });
}

Sakura::Component& MapsSettingsView::content() {
    return stack;
}

void MapsSettingsView::renderingChanged() {
    if (config.onRenderingChange) {
        config.onRenderingChange(config.rendering);
    }
}

std::string MapsSettingsView::cacheLabel() const {
    if (config.clearing) {
        return "Clearing downloaded maps.";
    }
    if (config.measuring) {
        return "Calculating size.";
    }
    if (!config.cacheAvailable || !config.cacheBytes.has_value()) {
        return "Saved maps are not available on this platform.";
    }
    const U64 bytes = config.cacheBytes.value();
    if (bytes == 0) {
        return "No maps downloaded yet.";
    }
    return jst::fmt::format("Using {} of storage.", SizeLabel(bytes));
}

std::string MapsSettingsView::SizeLabel(U64 bytes) {
    const F64 value = static_cast<F64>(bytes);
    if (value >= 1e9) {
        return jst::fmt::format("{:.1f} GB", value / 1e9);
    }
    if (value >= 1e6) {
        return jst::fmt::format("{:.1f} MB", value / 1e6);
    }
    if (value >= 1e3) {
        return jst::fmt::format("{:.0f} KB", value / 1e3);
    }
    return jst::fmt::format("{} bytes", bytes);
}

}  // namespace Jetstream
