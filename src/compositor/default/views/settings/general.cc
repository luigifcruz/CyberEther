#include "general.hh"

namespace Jetstream {

void GeneralSettingsView::update(Config config) {
    this->config = std::move(config);
    renderer = this->config.renderer.empty() ? "Metal" : this->config.renderer;

    header.update({
        .title = "General",
        .description = "Manage workspace layout, theme, and rendering.",
    });

    std::vector<Sakura::PillTabs::Item> themeItems;
    U64 themeIndex = 0;
    for (U64 index = 0; index < this->config.themes.size(); ++index) {
        const auto& theme = this->config.themes[index];
        themeItems.push_back({.icon = theme == "Light" ? ICON_LC_SUN : ICON_LC_MOON, .label = theme});
        if (theme == this->config.currentThemeKey) {
            themeIndex = index;
        }
    }
    themeTabs.update({
        .items = std::move(themeItems),
        .selected = themeIndex,
        .compact = true,
        .track = true,
        .onSelect = [this](U64 index) {
            if (index < this->config.themes.size() && this->config.onThemeChange) {
                this->config.onThemeChange(this->config.themes[index]);
            }
        },
    });

    scaleStepper.update({
        .options = scaleLabels(),
        .selected = scaleIndex(this->config.interfaceScale),
        .fallback = 10,
        .onSelect = [this](U64 index) {
            if (this->config.onInterfaceScaleChange) {
                this->config.onInterfaceScaleChange(scaleValues()[index]);
            }
        },
    });

    particlesToggle.update({
        .value = this->config.backgroundParticles,
        .onChange = [this](bool value) {
            if (this->config.onBackgroundParticlesChange) {
                this->config.onBackgroundParticlesChange(value);
            }
        },
    });

    tipsToggle.update({
        .value = this->config.showTips,
        .onChange = [this](bool value) {
            if (this->config.onShowTipsChange) {
                this->config.onShowTipsChange(value);
            }
        },
    });

    const auto renderers = rendererLabels();
    U64 rendererIndex = 0;
    std::vector<Sakura::PillTabs::Item> rendererItems;
    for (U64 index = 0; index < renderers.size(); ++index) {
        rendererItems.push_back({.label = renderers[index]});
        if (renderers[index] == renderer) {
            rendererIndex = index;
        }
    }
    rendererTabs.update({
        .items = std::move(rendererItems),
        .selected = rendererIndex,
        .compact = true,
        .track = true,
        .onSelect = [this](U64 index) {
            const auto labels = rendererLabels();
            if (index < labels.size() && this->config.onRendererChange) {
                this->config.onRendererChange(rendererValue(labels[index]));
            }
        },
    });

    appearanceGroup.update({
        .title = "Appearance",
        .description = "How the interface looks.",
        .rows = {
            {
                .label = "Theme",
                .description = "Select the color theme for the entire interface.",
                .control = &themeTabs,
            },
            {
                .label = "Interface Scale",
                .description = "Adjust the interface scale.",
                .control = &scaleStepper,
            },
            {
                .label = "Background Particles",
                .description = "Animate floating particles in the workspace background.",
                .control = &particlesToggle,
            },
            {
                .label = "Show Tips",
                .description = "Show a short tip when a flowgraph opens.",
                .control = &tipsToggle,
            },
        },
    });

    renderingGroup.update({
        .title = "Rendering",
        .description = "The graphics backend that draws the interface.",
        .rows = {{
            .label = "Graphics Renderer",
            .description = "Changing the renderer requires restarting CyberEther.",
            .control = &rendererTabs,
        }},
    });

    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &header},
            {.child = &appearanceGroup},
            {.child = &renderingGroup},
        },
    });
}

Sakura::Component& GeneralSettingsView::content() {
    return stack;
}

DeviceType GeneralSettingsView::rendererValue(const std::string& label) {
    if (label == "Metal") return DeviceType::Metal;
    if (label == "Vulkan") return DeviceType::Vulkan;
    if (label == "WebGPU") return DeviceType::WebGPU;
    return DeviceType::None;
}

std::vector<std::string> GeneralSettingsView::rendererLabels() {
    return {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
        "Metal",
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        "Vulkan",
#endif
#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
        "WebGPU",
#endif
    };
}

const std::vector<F32>& GeneralSettingsView::scaleValues() {
    static const std::vector<F32> values = [] {
        std::vector<F32> steps;
        for (U32 percent = 50; percent <= 300; percent += 5) {
            steps.push_back(static_cast<F32>(percent) / 100.0f);
        }
        return steps;
    }();
    return values;
}

std::vector<std::string> GeneralSettingsView::scaleLabels() {
    std::vector<std::string> labels;
    for (const F32 value : scaleValues()) {
        labels.push_back(jst::fmt::format("{:.0f}%", value * 100.0f));
    }
    return labels;
}

U64 GeneralSettingsView::scaleIndex(F32 scale) {
    const auto& values = scaleValues();
    U64 best = 0;
    for (U64 index = 0; index < values.size(); ++index) {
        if (std::abs(values[index] - scale) < std::abs(values[best] - scale)) {
            best = index;
        }
    }
    return best;
}

}  // namespace Jetstream
