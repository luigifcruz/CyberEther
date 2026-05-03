#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include "jetstream/instance_remote.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct RemoteSettingsPanel : public Sakura::Component {
    struct Config {
        bool started = false;
        std::string brokerUrl;
        Instance::Remote::CodecType codec = Instance::Remote::CodecType::H264;
        U32 framerate = 30;
        Instance::Remote::EncoderType encoder = Instance::Remote::EncoderType::Auto;
        bool autoJoinSessions = false;
        std::function<void(const std::string&)> onBrokerUrlChange;
        std::function<void(Instance::Remote::CodecType)> onCodecChange;
        std::function<void(U32)> onFramerateChange;
        std::function<void(Instance::Remote::EncoderType)> onEncoderChange;
        std::function<void(bool)> onAutoJoinSessionsChange;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "RemoteTitle",
            .str = "Remote",
            .scale = 1.2f,
        });

        description.update({
            .id = "RemoteDescription",
            .str = "Configure default remote streaming parameters.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "RemoteHeaderDivider",
        });

        activeWarning.update({
            .id = "RemoteActiveWarning",
            .str = ICON_FA_TOWER_BROADCAST " Remote streaming is currently active.",
            .tone = Sakura::Text::Tone::Warning,
        });

        activeDescription.update({
            .id = "RemoteActiveDescription",
            .str = "Changes here update the defaults for the next session. Use the remote streaming panel to manage the live session.",
            .tone = Sakura::Text::Tone::Disabled,
            .wrapped = true,
        });

        activeSpacing.update({
            .id = "RemoteActiveSpacing",
        });

        brokerField.update({
            .id = "RemoteBrokerField",
            .label = "Broker URL",
            .description = "The WebSocket broker endpoint that coordinates remote client connections.",
        });

        brokerInput.update({
            .id = "##app-settings-broker-url",
            .value = this->config.brokerUrl,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                if (this->config.onBrokerUrlChange) {
                    this->config.onBrokerUrlChange(value);
                }
            },
        });

        codecField.update({
            .id = "RemoteCodecField",
            .label = "Video Codec",
            .description = "The video compression format used for streaming.",
        });

        codecCombo.update({
            .id = "##app-settings-codec",
            .options = {
                "H264",
                "AV1",
                "VP8",
                "VP9",
            },
            .value = codecLabel(this->config.codec),
            .onChange = [this](const std::string& label) {
                if (this->config.onCodecChange) {
                    this->config.onCodecChange(codecValue(label));
                }
            },
        });

        framerateField.update({
            .id = "RemoteFramerateField",
            .label = "Framerate",
            .description = "Target frame rate for the outgoing video stream.",
        });

        framerateCombo.update({
            .id = "##app-settings-framerate",
            .options = {
                "15 FPS",
                "30 FPS",
                "60 FPS",
                "120 FPS",
            },
            .value = framerateLabel(this->config.framerate),
            .onChange = [this](const std::string& label) {
                if (this->config.onFramerateChange) {
                    this->config.onFramerateChange(framerateValue(label));
                }
            },
        });

        encoderField.update({
            .id = "RemoteEncoderField",
            .label = "Encoder",
            .description = "The hardware or software encoder used to compress the video stream.",
        });

        encoderCombo.update({
            .id = "##app-settings-encoder",
            .options = {
                "Auto",
                "Software",
                "NVENC",
                "V4L2",
            },
            .value = encoderLabel(this->config.encoder),
            .onChange = [this](const std::string& label) {
                if (this->config.onEncoderChange) {
                    this->config.onEncoderChange(encoderValue(label));
                }
            },
        });

        approvalField.update({
            .id = "RemoteApprovalField",
            .label = "Client Approval",
            .description = "Whether new remote clients are accepted automatically or require manual approval.",
            .divider = false,
        });

        approvalCombo.update({
            .id = "##app-settings-client-approval",
            .options = {
                "Manual Approval",
                "Auto Approve",
            },
            .value = this->config.autoJoinSessions ? "Auto Approve" : "Manual Approval",
            .onChange = [this](const std::string& label) {
                if (this->config.onAutoJoinSessionsChange) {
                    this->config.onAutoJoinSessionsChange(label == "Auto Approve");
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        if (config.started) {
            activeWarning.render(ctx);
            activeDescription.render(ctx);
            activeSpacing.render(ctx);
        }

        brokerField.render(ctx, [&](const Sakura::Context& ctx) {
            brokerInput.render(ctx);
        });

        codecField.render(ctx, [&](const Sakura::Context& ctx) {
            codecCombo.render(ctx);
        });

        framerateField.render(ctx, [&](const Sakura::Context& ctx) {
            framerateCombo.render(ctx);
        });

        encoderField.render(ctx, [&](const Sakura::Context& ctx) {
            encoderCombo.render(ctx);
        });

        approvalField.render(ctx, [&](const Sakura::Context& ctx) {
            approvalCombo.render(ctx);
        });
    }

 private:
    static std::string codecLabel(Instance::Remote::CodecType codec) {
        switch (codec) {
            case Instance::Remote::CodecType::H264: return "H264";
            case Instance::Remote::CodecType::AV1: return "AV1";
            case Instance::Remote::CodecType::VP8: return "VP8";
            case Instance::Remote::CodecType::VP9: return "VP9";
        }
        return "H264";
    }

    static Instance::Remote::CodecType codecValue(const std::string& label) {
        if (label == "H264") return Instance::Remote::CodecType::H264;
        if (label == "AV1") return Instance::Remote::CodecType::AV1;
        if (label == "VP8") return Instance::Remote::CodecType::VP8;
        if (label == "VP9") return Instance::Remote::CodecType::VP9;
        return Instance::Remote::CodecType::H264;
    }

    static std::string framerateLabel(U32 framerate) {
        switch (framerate) {
            case 15: return "15 FPS";
            case 30: return "30 FPS";
            case 60: return "60 FPS";
            case 120: return "120 FPS";
        }
        return "30 FPS";
    }

    static U32 framerateValue(const std::string& label) {
        if (label == "15 FPS") return 15;
        if (label == "30 FPS") return 30;
        if (label == "60 FPS") return 60;
        if (label == "120 FPS") return 120;
        return 30;
    }

    static std::string encoderLabel(Instance::Remote::EncoderType encoder) {
        switch (encoder) {
            case Instance::Remote::EncoderType::Auto: return "Auto";
            case Instance::Remote::EncoderType::Software: return "Software";
            case Instance::Remote::EncoderType::NVENC: return "NVENC";
            case Instance::Remote::EncoderType::V4L2: return "V4L2";
        }
        return "Auto";
    }

    static Instance::Remote::EncoderType encoderValue(const std::string& label) {
        if (label == "Auto") return Instance::Remote::EncoderType::Auto;
        if (label == "Software") return Instance::Remote::EncoderType::Software;
        if (label == "NVENC") return Instance::Remote::EncoderType::NVENC;
        if (label == "V4L2") return Instance::Remote::EncoderType::V4L2;
        return Instance::Remote::EncoderType::Auto;
    }

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Text activeWarning;
    Sakura::Text activeDescription;
    Sakura::Spacing activeSpacing;
    Sakura::SettingField brokerField;
    Sakura::SettingField codecField;
    Sakura::SettingField framerateField;
    Sakura::SettingField encoderField;
    Sakura::SettingField approvalField;
    Sakura::TextInput brokerInput;
    Sakura::Combo codecCombo;
    Sakura::Combo framerateCombo;
    Sakura::Combo encoderCombo;
    Sakura::Combo approvalCombo;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH
