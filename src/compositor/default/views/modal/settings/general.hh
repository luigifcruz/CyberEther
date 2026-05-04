#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_GENERAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_GENERAL_HH

#include "jetstream/memory/types.hh"
#include "jetstream/render/sakura/sakura.hh"

#include <functional>
#include <string>
#include <vector>

namespace Jetstream {

struct GeneralSettingsPanel : public Sakura::Component {
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
        interfaceScalePreview = this->config.interfaceScale;
        frameRateLimit = framerateLabel(this->config.framerate);

        title.update({
            .id = "GeneralTitle",
            .str = "General",
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
            .description = "Changing the interface scale requires restarting CyberEther.",
        });

        scaleSlider.update({
            .id = "##scale",
            .value = interfaceScalePreview,
            .min = 0.5f,
            .max = 2.0f,
            .format = "%.2fx",
            .onChange = [this](F32 value) {
                interfaceScalePreview = value;
                if (this->config.onInterfaceScaleChange) {
                    this->config.onInterfaceScaleChange(value);
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
                "Metal",
                "Vulkan",
                "WebGPU",
            },
            .value = renderer,
            .onChange = [this](const std::string& value) {
                renderer = value;
                if (this->config.onRendererChange) {
                    this->config.onRendererChange(rendererValue(value));
                }
            },
        });

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
            scaleSlider.render(ctx);
        });

        rendererField.render(ctx, [&](const Sakura::Context& ctx) {
            rendererCombo.render(ctx);
        });

        frameRateField.render(ctx, [&](const Sakura::Context& ctx) {
            frameRateCombo.render(ctx);
        });

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
    Sakura::Combo rendererCombo;
    Sakura::Combo frameRateCombo;
    Sakura::SliderFloat scaleSlider;
    Sakura::Checkbox infoPanelCheckbox;
    Sakura::Checkbox particlesCheckbox;
    std::string renderer;
    std::string frameRateLimit = "Auto";
    F32 interfaceScalePreview = 0.0f;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_GENERAL_HH
