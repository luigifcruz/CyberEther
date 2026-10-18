#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_GENERAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_GENERAL_HH

#include "jetstream/memory/types.hh"
#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace Jetstream {

struct GeneralSettingsPanel {
    struct Config {
        std::vector<std::string> themes;
        std::string currentThemeKey;
        F32 interfaceScale = 1.0f;
        std::string renderer;
        bool infoPanelEnabled = false;
        bool backgroundParticles = false;
        std::function<void(const std::string&)> onThemeChange;
        std::function<void(F32)> onInterfaceScaleChange;
        std::function<void(DeviceType)> onRendererChange;
        std::function<void(bool)> onInfoPanelChange;
        std::function<void(bool)> onBackgroundParticlesChange;
    };

    void update(Config config) {
        this->config = std::move(config);
        renderer = this->config.renderer.empty() ? "Metal" : this->config.renderer;
        interfaceScale = scaleLabel(this->config.interfaceScale);

        using Text = Sakura::Text;

        title.update({
            .str = "General",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .str = "Manage workspace layout, theme, and rendering.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({});

        themeField.update({
            .label = "Theme",
            .description = "Select the color theme for the entire interface.",
            .control = &themeCombo,
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
        });

        scaleField.update({
            .label = "Interface Scale",
            .description = "Adjust the interface scale.",
            .control = &scaleCombo,
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
        });

        rendererField.update({
            .label = "Graphics Renderer",
            .description = "Changing the renderer requires restarting CyberEther.",
            .control = &rendererCombo,
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
        });

        infoPanelField.update({
            .description = "Display the info panel alongside the main workspace.",
            .control = &infoPanelCheckbox,
        });

        infoPanelCheckbox.update({
            .label = "Show info panel",
            .value = this->config.infoPanelEnabled,
            .onChange = [this](bool value) {
                if (this->config.onInfoPanelChange) {
                    this->config.onInfoPanelChange(value);
                }
            },
        });

        particlesField.update({
            .description = "Animate floating particles in the workspace background.",
            .control = &particlesCheckbox,
        });

        particlesCheckbox.update({
            .label = "Background particles",
            .value = this->config.backgroundParticles,
            .onChange = [this](bool value) {
                if (this->config.onBackgroundParticlesChange) {
                    this->config.onBackgroundParticlesChange(value);
                }
            },
        });

        stack.update({
            .items = {
                {.child = &title},
                {.child = &description},
                {.child = &divider},
                {.child = &themeField},
                {.child = &scaleField},
                {.child = &rendererField},
                {.child = &infoPanelField},
                {.child = &particlesField},
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
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Combo themeCombo;
    Sakura::Combo scaleCombo;
    Sakura::Combo rendererCombo;
    Sakura::Checkbox infoPanelCheckbox;
    Sakura::Checkbox particlesCheckbox;
    Sakura::SettingField themeField;
    Sakura::SettingField scaleField;
    Sakura::SettingField rendererField;
    Sakura::SettingField infoPanelField;
    Sakura::SettingField particlesField;
    Sakura::Stack stack;
    std::string renderer;
    std::string interfaceScale = "1.00x";
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_GENERAL_HH
