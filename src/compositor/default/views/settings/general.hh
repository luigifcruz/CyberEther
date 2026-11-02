#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_GENERAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_GENERAL_HH

#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/memory/types.hh"
#include "jetstream/render/sakura/base.hh"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace Jetstream {

struct GeneralSettingsView {
    struct Config {
        std::vector<std::string> themes;
        std::string currentThemeKey;
        F32 interfaceScale = 1.0f;
        std::string renderer;
        bool backgroundParticles = false;
        bool showTips = true;
        Sakura::Callback<void(const std::string&)> onThemeChange;
        Sakura::Callback<void(F32)> onInterfaceScaleChange;
        Sakura::Callback<void(DeviceType)> onRendererChange;
        Sakura::Callback<void(bool)> onBackgroundParticlesChange;
        Sakura::Callback<void(bool)> onShowTipsChange;
    };

    void update(Config config) {
        this->config = std::move(config);
        renderer = this->config.renderer.empty() ? "Metal" : this->config.renderer;
        interfaceScale = scaleLabel(this->config.interfaceScale);

        header.update({
            .title = "General",
            .description = "Manage workspace layout, theme, and rendering.",
        });

        themeCombo.update({
            .id = "settings:theme",
            .options = this->config.themes,
            .value = this->config.currentThemeKey,
            .onChange = [this](const std::string& themeKey) {
                if (this->config.onThemeChange) {
                    this->config.onThemeChange(themeKey);
                }
            },
            .borderColorKey = "border",
        });

        scaleCombo.update({
            .id = "settings:scale",
            .options = {
                "0.50x",
                "0.75x",
                "1.00x",
                "1.15x",
                "1.25x",
                "1.50x",
                "2.00x",
                "3.00x",
            },
            .value = interfaceScale,
            .onChange = [this](const std::string& value) {
                interfaceScale = value;
                if (this->config.onInterfaceScaleChange) {
                    this->config.onInterfaceScaleChange(scaleValue(value));
                }
            },
            .borderColorKey = "border",
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

        rendererCombo.update({
            .id = "settings:renderer",
            .options = {
#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
                "Metal",
#endif
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
                "Vulkan",
#endif
#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
                "WebGPU",
#endif
            },
            .value = renderer,
            .onChange = [this](const std::string& value) {
                renderer = value;
                if (this->config.onRendererChange) {
                    this->config.onRendererChange(rendererValue(value));
                }
            },
            .borderColorKey = "border",
        });

        appearanceGroup.update({
            .title = "Appearance",
            .description = "How the interface looks.",
            .rows = {
                {
                    .label = "Theme",
                    .description = "Select the color theme for the entire interface.",
                    .control = &themeCombo,
                },
                {
                    .label = "Interface Scale",
                    .description = "Adjust the interface scale.",
                    .control = &scaleCombo,
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
                .control = &rendererCombo,
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

    Sakura::Component& content() {
        return stack;
    }

 private:
    static DeviceType rendererValue(const std::string& label) {
        if (label == "Metal") return DeviceType::Metal;
        if (label == "Vulkan") return DeviceType::Vulkan;
        if (label == "WebGPU") return DeviceType::WebGPU;
        return DeviceType::None;
    }

    static std::string scaleLabel(F32 scale) {
        if (scale == 0.5f) return "0.50x";
        if (scale == 0.75f) return "0.75x";
        if (scale == 1.0f) return "1.00x";
        if (scale == 1.15f) return "1.15x";
        if (scale == 1.25f) return "1.25x";
        if (scale == 1.5f) return "1.50x";
        if (scale == 2.0f) return "2.00x";
        if (scale == 3.0f) return "3.00x";

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << scale << "x";
        return stream.str();
    }

    static F32 scaleValue(const std::string& label) {
        if (label == "0.50x") return 0.5f;
        if (label == "0.75x") return 0.75f;
        if (label == "1.00x") return 1.0f;
        if (label == "1.15x") return 1.15f;
        if (label == "1.25x") return 1.25f;
        if (label == "1.50x") return 1.5f;
        if (label == "2.00x") return 2.0f;
        if (label == "3.00x") return 3.0f;
        return 1.0f;
    }

    Config config;
    SettingsHeader header;
    Sakura::Combo themeCombo;
    Sakura::Combo scaleCombo;
    Sakura::Combo rendererCombo;
    Sakura::Toggle particlesToggle;
    Sakura::Toggle tipsToggle;
    SectionGroup appearanceGroup;
    SectionGroup renderingGroup;
    Sakura::Stack stack;
    std::string renderer;
    std::string interfaceScale = "1.00x";
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_GENERAL_HH
