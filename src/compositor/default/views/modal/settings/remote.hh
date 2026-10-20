#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"

#include "jetstream/instance_remote.hh"

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteSettingsPanel {
    struct Section : public Sakura::Stack {
        struct Config {
            std::string title;
            std::string description;
            std::string warning;
            std::vector<Sakura::Component*> fields;
        };

        void update(Config config) {
            using Text = Sakura::Text;

            titleText.update({
                .str = config.title,
                .font = Text::Font::Bold,
                .scale = 1.1f,
            });
            descriptionText.update({
                .str = config.description,
                .tone = Text::Tone::Secondary,
                .wrapped = true,
            });
            warningText.update({
                .str = config.warning.empty() ? "" : ICON_FA_TRIANGLE_EXCLAMATION " " + config.warning,
                .tone = Text::Tone::Warning,
                .wrapped = true,
            });
            spacing.update({});
            trailing.update({.lines = 2});

            std::vector<Item> fieldItems;
            fieldItems.reserve(config.fields.size());
            for (auto* field : config.fields) {
                fieldItems.push_back({.child = field});
            }
            fields.update({.items = std::move(fieldItems)});
            card.update({
                .content = &fields,
                .padding = 16.0f,
                .rounding = 8.0f,
                .border = true,
                .colorKey = "transparent",
                .hoveredColorKey = "transparent",
            });

            std::vector<Item> items = {
                {.child = &titleText},
                {.child = &descriptionText},
            };
            if (!config.warning.empty()) {
                items.push_back({.child = &warningText});
            }
            items.push_back({.child = &spacing});
            items.push_back({.child = &card});
            items.push_back({.child = &trailing});
            Stack::update({.items = std::move(items)});
        }

     private:
        Sakura::Text titleText;
        Sakura::Text descriptionText;
        Sakura::Text warningText;
        Sakura::Spacer spacing;
        Sakura::Stack fields;
        Sakura::Div card;
        Sakura::Spacer trailing;
    };

    struct Config {
        bool started = false;
        Instance::Remote::Config::Broker broker;
        Instance::Remote::CodecType codec = Instance::Remote::CodecType::H264;
        U32 framerate = 30;
        Instance::Remote::EncoderType encoder = Instance::Remote::EncoderType::Auto;
        std::vector<Instance::Remote::EncoderType> available;
        bool autoApprove = false;
        Instance::Remote::TransportType transport = Instance::Remote::TransportType::Auto;
        U16 minPort = 0;
        U16 maxPort = 0;
        std::string mediaAddress;
        std::function<void(const Instance::Remote::Config::Broker&)> onBrokerChange;
        std::function<void(const std::string&, std::vector<std::string>, std::function<void(std::string)>)> onBrowseFile;
        std::function<void(Instance::Remote::CodecType)> onCodecChange;
        std::function<void(U32)> onFramerateChange;
        std::function<void(Instance::Remote::EncoderType)> onEncoderChange;
        std::function<void(bool)> onAutoApproveChange;
        std::function<void(Instance::Remote::TransportType)> onTransportChange;
        std::function<void(U16, U16)> onPortRangeChange;
        std::function<void(const std::string&)> onMediaAddressChange;
    };

    void update(Config config) {
        using Text = Sakura::Text;
        using TextInput = Sakura::TextInput;

        this->config = std::move(config);

        title.update({
            .str = "Remote",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .str = "Configure default remote streaming parameters.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({});

        activeWarning.update({
            .str = ICON_FA_TOWER_BROADCAST " Remote streaming is currently active.",
            .tone = Text::Tone::Warning,
            .scale = 1.05f,
        });

        activeDescription.update({
            .str = "Updates here take effect after the current session ends.",
            .tone = Text::Tone::Disabled,
            .wrapped = true,
        });

        activeSpacing.update({});
        activeTrailing.update({});

        approvalCombo.update({
            .id = "settings:client-approval",
            .options = {
                "Manual Approval",
                "Auto Approve",
            },
            .value = this->config.autoApprove ? "Auto Approve" : "Manual Approval",
            .onChange = [this](const std::string& label) {
                if (this->config.onAutoApproveChange) {
                    this->config.onAutoApproveChange(label == "Auto Approve");
                }
            },
        });

        approvalField.update({
            .label = "Viewer Approval",
            .description = "Whether new viewers join automatically or wait for manual approval.",
            .control = &approvalCombo,
            .divider = false,
        });

        framerateCombo.update({
            .id = "settings:framerate",
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

        framerateField.update({
            .label = "Video Framerate",
            .description = "Target framerate for the outgoing video stream.",
            .control = &framerateCombo,
        });

        codecCombo.update({
            .id = "settings:codec",
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

        codecField.update({
            .label = "Video Codec",
            .description = "The video compression format used for streaming.",
            .control = &codecCombo,
        });

        encoderCombo.update({
            .id = "settings:encoder",
            .options = encoderOptions(this->config.available),
            .value = encoderLabel(this->config.encoder, this->config.available),
            .disabled = this->config.available.empty(),
            .onChange = [this](const std::string& label) {
                if (this->config.onEncoderChange) {
                    this->config.onEncoderChange(encoderValue(label));
                }
            },
        });

        encoderField.update({
            .label = "Video Encoder",
            .description = "The hardware or software encoder used to compress the video stream.",
            .control = &encoderCombo,
            .divider = false,
        });

        brokerAddressInput.update({
            .value = this->config.broker.address,
            .submit = TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.address = value.empty() ? Instance::Remote::Config::Broker{}.address : value;
                changeBroker(broker);
            },
        });

        brokerAddressField.update({
            .label = "Broker Address",
            .description = "The network address the broker listens on. Use 0.0.0.0 to accept viewers on every interface.",
            .control = &brokerAddressInput,
        });

        brokerPortInput.update({
            .value = jst::fmt::format("{}", this->config.broker.port),
            .submit = TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                U16 port = 0;
                if (portValue(value, port)) {
                    auto broker = this->config.broker;
                    broker.port = port;
                    changeBroker(broker);
                }
            },
        });

        brokerPortField.update({
            .label = "Broker Port",
            .description = "The port that serves the remote viewer page and its connections.",
            .control = &brokerPortInput,
        });

        brokerPublicUrlInput.update({
            .value = this->config.broker.publicUrl,
            .hint = "Automatic",
            .submit = TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                if (!value.empty() && !IsRemotePublicUrlValid(value)) {
                    return;
                }
                auto broker = this->config.broker;
                broker.publicUrl = value;
                changeBroker(broker);
            },
        });

        brokerPublicUrlField.update({
            .label = "Broker Public URL",
            .description = "The address viewers use to reach this machine, written as scheme, host, and port only. Leave empty to detect the local network address.",
            .control = &brokerPublicUrlInput,
        });

        brokerCertificateInput.update({
            .value = this->config.broker.certificate,
            .hint = "None",
            .submit = TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.certificate = value;
                changeBroker(broker);
            },
        });

        brokerCertificateBrowse.update({
            .str = "Browse File",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                browseFile(this->config.broker.certificate, {"pem", "crt", "cer"}, [this](std::string path) {
                    auto broker = this->config.broker;
                    broker.certificate = std::move(path);
                    changeBroker(broker);
                });
            },
        });

        brokerCertificateControls.update({
            .items = {
                {.child = &brokerCertificateInput},
                {.child = &brokerCertificateBrowse},
            },
        });

        brokerCertificateField.update({
            .label = "Broker TLS Certificate",
            .description = "Path to a PEM certificate that enables HTTPS. Leave empty to serve plain HTTP.",
            .control = &brokerCertificateControls,
        });

        brokerKeyInput.update({
            .value = this->config.broker.key,
            .hint = "None",
            .submit = TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.key = value;
                changeBroker(broker);
            },
        });

        brokerKeyBrowse.update({
            .str = "Browse File",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                browseFile(this->config.broker.key, {"pem", "key"}, [this](std::string path) {
                    auto broker = this->config.broker;
                    broker.key = std::move(path);
                    changeBroker(broker);
                });
            },
        });

        brokerKeyControls.update({
            .items = {
                {.child = &brokerKeyInput},
                {.child = &brokerKeyBrowse},
            },
        });

        brokerKeyField.update({
            .label = "Broker TLS Key",
            .description = "Path to the PEM private key that matches the certificate.",
            .control = &brokerKeyControls,
            .divider = false,
        });

        transportCombo.update({
            .id = "settings:transport",
            .options = transportOptions(),
            .value = GetRemoteTransportPrettyName(this->config.transport),
            .onChange = [this](const std::string& label) {
                if (this->config.onTransportChange) {
                    this->config.onTransportChange(transportValue(label));
                }
            },
        });

        transportField.update({
            .label = "Media Transport",
            .description = "The network protocols offered for the media connection. TCP helps on networks that block UDP.",
            .control = &transportCombo,
        });

        portRangeInput.update({
            .value = GetRemotePortRangeName(this->config.minPort, this->config.maxPort),
            .hint = "Any",
            .submit = TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                U16 minPort = 0;
                U16 maxPort = 0;
                if (this->config.onPortRangeChange && ParseRemotePortRange(value, minPort, maxPort)) {
                    this->config.onPortRangeChange(minPort, maxPort);
                }
            },
        });

        portRangeField.update({
            .label = "Media Port Range",
            .description = "The local ports used for the media connection, written as MIN-MAX. Leave empty to let the system choose.",
            .control = &portRangeInput,
        });

        mediaAddressInput.update({
            .value = this->config.mediaAddress,
            .hint = "Automatic",
            .submit = TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                if (this->config.onMediaAddressChange) {
                    this->config.onMediaAddressChange(value);
                }
            },
        });

        mediaAddressField.update({
            .label = "Media Address",
            .description = "The only local IP address offered for the media connection. Use 127.0.0.1 when tunneling over SSH. Leave empty to offer every address.",
            .control = &mediaAddressInput,
            .divider = false,
        });

        accessSection.update({
            .title = "Access",
            .description = "Who can join a remote session.",
            .fields = {&approvalField},
        });
        videoSection.update({
            .title = "Video",
            .description = "How the interface is encoded for viewers.",
            .fields = {&framerateField, &codecField, &encoderField},
        });
        brokerSection.update({
            .title = "Broker",
            .description = "The server on this machine that hosts the viewer page and connects viewers.",
            .warning = "Do not expose the broker directly to the internet.",
            .fields = {
                &brokerAddressField,
                &brokerPortField,
                &brokerPublicUrlField,
                &brokerCertificateField,
                &brokerKeyField,
            },
        });
        mediaSection.update({
            .title = "Media",
            .description = "Network settings for the video connection.",
            .fields = {&transportField, &portRangeField, &mediaAddressField},
        });

        activeContent.update({
            .items = {
                {.child = &activeWarning},
                {.child = &activeSpacing},
                {.child = &activeDescription},
            },
        });
        activeCard.update({
            .content = &activeContent,
            .padding = 16.0f,
            .rounding = 8.0f,
            .border = true,
        });
        std::vector<Sakura::Stack::Item> items = {
            {.child = &title},
            {.child = &description},
            {.child = &divider},
        };
        if (this->config.started) {
            items.push_back({.child = &activeCard});
            items.push_back({.child = &activeTrailing});
        }
        items.push_back({.child = &accessSection});
        items.push_back({.child = &videoSection});
        items.push_back({.child = &brokerSection});
        items.push_back({.child = &mediaSection});
        stack.update({.items = std::move(items)});
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    void browseFile(const std::string& path,
                    std::vector<std::string> extensions,
                    std::function<void(std::string)> onSelect) const {
        if (config.onBrowseFile) {
            config.onBrowseFile(path, std::move(extensions), std::move(onSelect));
        }
    }

    void changeBroker(const Instance::Remote::Config::Broker& broker) const {
        if (config.onBrokerChange) {
            config.onBrokerChange(broker);
        }
    }

    static bool portValue(const std::string& label, U16& port) {
        if (label.empty() || label.size() > 5 || !std::all_of(label.begin(), label.end(), [](unsigned char c) {
            return std::isdigit(c);
        })) {
            return false;
        }
        const U64 value = std::stoull(label);
        if (value > 65535) {
            return false;
        }
        port = static_cast<U16>(value);
        return true;
    }

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

    static std::string encoderLabel(Instance::Remote::EncoderType encoder,
                                    const std::vector<Instance::Remote::EncoderType>& encoders) {
        if (encoders.empty()) {
            return "Unavailable";
        }
        return GetRemoteEncoderPrettyName(encoder);
    }

    static std::vector<std::string> encoderOptions(const std::vector<Instance::Remote::EncoderType>& encoders) {
        std::vector<std::string> options;
        options.reserve(encoders.size());
        for (const auto encoder : encoders) {
            options.push_back(GetRemoteEncoderPrettyName(encoder));
        }
        return options;
    }

    static std::vector<std::string> transportOptions() {
        std::vector<std::string> options;
        options.reserve(RemoteTransportTypes.size());
        for (const auto transport : RemoteTransportTypes) {
            options.push_back(GetRemoteTransportPrettyName(transport));
        }
        return options;
    }

    static Instance::Remote::TransportType transportValue(const std::string& label) {
        for (const auto transport : RemoteTransportTypes) {
            if (label == GetRemoteTransportPrettyName(transport)) {
                return transport;
            }
        }
        return Instance::Remote::TransportType::Auto;
    }

    static Instance::Remote::EncoderType encoderValue(const std::string& label) {
        try {
            return StringToRemoteEncoder(label);
        } catch (const Result&) {
            return Instance::Remote::EncoderType::Auto;
        }
    }

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Text activeWarning;
    Sakura::Text activeDescription;
    Sakura::Spacer activeSpacing;
    Sakura::Spacer activeTrailing;
    Sakura::Stack activeContent;
    Sakura::Div activeCard;
    Sakura::Combo approvalCombo;
    Sakura::SettingField approvalField;
    Sakura::Combo framerateCombo;
    Sakura::SettingField framerateField;
    Sakura::Combo codecCombo;
    Sakura::SettingField codecField;
    Sakura::Combo encoderCombo;
    Sakura::SettingField encoderField;
    Sakura::TextInput brokerAddressInput;
    Sakura::SettingField brokerAddressField;
    Sakura::TextInput brokerPortInput;
    Sakura::SettingField brokerPortField;
    Sakura::TextInput brokerPublicUrlInput;
    Sakura::SettingField brokerPublicUrlField;
    Sakura::TextInput brokerCertificateInput;
    Sakura::Button brokerCertificateBrowse;
    Sakura::Stack brokerCertificateControls;
    Sakura::SettingField brokerCertificateField;
    Sakura::TextInput brokerKeyInput;
    Sakura::Button brokerKeyBrowse;
    Sakura::Stack brokerKeyControls;
    Sakura::SettingField brokerKeyField;
    Sakura::Combo transportCombo;
    Sakura::SettingField transportField;
    Sakura::TextInput portRangeInput;
    Sakura::SettingField portRangeField;
    Sakura::TextInput mediaAddressInput;
    Sakura::SettingField mediaAddressField;
    Section accessSection;
    Section videoSection;
    Section brokerSection;
    Section mediaSection;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH
