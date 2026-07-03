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
        U64 framerate = 60;
        bool infoPanelEnabled = false;
        bool backgroundParticles = false;
        std::function<void(const std::string&)> onThemeChange;
        std::function<void(F32)> onInterfaceScaleChange;
        std::function<void(DeviceType)> onRendererChange;
        std::function<void(U64)> onFramerateChange;
        std::function<void(bool)> onInfoPanelChange;
        std::function<void(bool)> onBackgroundParticlesChange;
    };

    void update(Config config) {
        this->config = std::move(config);
        renderer = this->config.renderer.empty() ? "Metal" : this->config.renderer;
        interfaceScale = scaleLabel(this->config.interfaceScale);
        frameRateLimit = framerateLabel(this->config.framerate);

        title.update({
            .id = "GeneralTitle",
            .str = "General",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .id = "GeneralDescription",
            .str = "Manage workspace layout, theme, and rendering.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "GeneralHeaderDivider",
        });

        themeField.update({
            .id = "ThemeField",
            .label = "Theme",
            .description = "Select the color theme for the entire interface.",
        });

        themeCombo.update({
            .id = "##theme",
            .options = this->config.themes,
            .value = this->config.currentThemeKey,
            .onChange = [this](const std::string& themeKey) {
                if (this->config.onThemeChange) {
                    this->config.onThemeChange(themeKey);
                }
            },
        });

        scaleField.update({
            .id = "ScaleField",
            .label = "Interface Scale",
            .description = "Adjust the interface scale.",
        });

        scaleCombo.update({
            .id = "##scale",
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
            .id = "RendererField",
            .label = "Graphics Renderer",
            .description = "Changing the renderer requires restarting CyberEther.",
        });

        rendererCombo.update({
            .id = "##renderer",
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

// TODO: Implement framerate limit.
#if 0
        frameRateField.update({
            .id = "FrameRateField",
            .label = "Frame Rate Limit",
            .description = "Changing the frame rate limit requires restarting CyberEther.",
        });

        frameRateCombo.update({
            .id = "##framerate",
            .options = {
                "5 FPS",
                "10 FPS",
                "15 FPS",
                "30 FPS",
                "60 FPS",
                "120 FPS",
                "240 FPS",
            },
            .value = frameRateLimit,
            .onChange = [this](const std::string& value) {
                frameRateLimit = value;
                if (this->config.onFramerateChange) {
                    this->config.onFramerateChange(framerateValue(value));
                }
            },
        });
#endif

        infoPanelField.update({
            .id = "InfoPanelField",
            .description = "Display the info panel alongside the main workspace.",
        });

        infoPanelCheckbox.update({
            .id = "InfoPanelCheckbox",
            .label = "Show info panel",
            .value = this->config.infoPanelEnabled,
            .onChange = [this](bool value) {
                if (this->config.onInfoPanelChange) {
                    this->config.onInfoPanelChange(value);
                }
            },
        });

        particlesField.update({
            .id = "ParticlesField",
            .description = "Animate floating particles in the workspace background.",
        });

        particlesCheckbox.update({
            .id = "ParticlesCheckbox",
            .label = "Background particles",
            .value = this->config.backgroundParticles,
            .onChange = [this](bool value) {
                if (this->config.onBackgroundParticlesChange) {
                    this->config.onBackgroundParticlesChange(value);
                }
            },
        });

    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        themeField.render(ctx, [&](const Sakura::Context& ctx) {
            themeCombo.render(ctx);
        });

        scaleField.render(ctx, [&](const Sakura::Context& ctx) {
            scaleCombo.render(ctx);
        });

        rendererField.render(ctx, [&](const Sakura::Context& ctx) {
            rendererCombo.render(ctx);
        });

#if 0
        frameRateField.render(ctx, [&](const Sakura::Context& ctx) {
            frameRateCombo.render(ctx);
        });
#endif

        infoPanelField.render(ctx, [&](const Sakura::Context& ctx) {
            infoPanelCheckbox.render(ctx);
        });

        particlesField.render(ctx, [&](const Sakura::Context& ctx) {
            particlesCheckbox.render(ctx);
        });
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

    static std::string framerateLabel(U64 framerate) {
        switch (framerate) {
            case 5: return "5 FPS";
            case 10: return "10 FPS";
            case 15: return "15 FPS";
            case 30: return "30 FPS";
            case 60: return "60 FPS";
            case 120: return "120 FPS";
            case 240: return "240 FPS";
        }
        return "60 FPS";
    }

    static U64 framerateValue(const std::string& label) {
        if (label == "5 FPS") return 5;
        if (label == "10 FPS") return 10;
        if (label == "15 FPS") return 15;
        if (label == "30 FPS") return 30;
        if (label == "60 FPS") return 60;
        if (label == "120 FPS") return 120;
        if (label == "240 FPS") return 240;
        return 60;
    }

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::SettingField themeField;
    Sakura::SettingField scaleField;
    Sakura::SettingField rendererField;
    Sakura::SettingField frameRateField;
    Sakura::SettingField infoPanelField;
    Sakura::SettingField particlesField;
    Sakura::Combo themeCombo;
    Sakura::Combo scaleCombo;
    Sakura::Combo rendererCombo;
    Sakura::Combo frameRateCombo;
    Sakura::Checkbox infoPanelCheckbox;
    Sakura::Checkbox particlesCheckbox;
    std::string renderer;
    std::string interfaceScale = "1.00x";
    std::string frameRateLimit = "Auto";
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_GENERAL_HH
