#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include "jetstream/instance_remote.hh"

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

namespace Jetstream {

struct RemoteSettingsPanel {
    struct Section {
        void update(const std::string& id,
                    const std::string& title,
                    const std::string& description,
                    const std::string& warning = "") {
            titleText.update({
                .id = id + "Title",
                .str = title,
                .font = Sakura::Text::Font::Bold,
                .scale = 1.1f,
            });
            descriptionText.update({
                .id = id + "Description",
                .str = description,
                .tone = Sakura::Text::Tone::Secondary,
                .wrapped = true,
            });
            warningText.update({
                .id = id + "Warning",
                .str = warning.empty() ? "" : ICON_FA_TRIANGLE_EXCLAMATION " " + warning,
                .tone = Sakura::Text::Tone::Warning,
                .wrapped = true,
            });
            hasWarning = !warning.empty();
            card.update({
                .id = id + "Card",
                .padding = 16.0f,
                .rounding = 8.0f,
                .border = true,
                .scrollbar = false,
                .mouseScroll = false,
                .colorKey = "transparent",
                .hoveredColorKey = "transparent",
            });
            spacing.update({
                .id = id + "Spacing",
            });
        }

        void render(const Sakura::Context& ctx, const std::function<void(const Sakura::Context&)>& fields) const {
            titleText.render(ctx);
            descriptionText.render(ctx);
            if (hasWarning) {
                warningText.render(ctx);
            }
            spacing.render(ctx);
            card.render(ctx, fields);
            spacing.render(ctx);
            spacing.render(ctx);
        }

     private:
        Sakura::Text titleText;
        Sakura::Text descriptionText;
        Sakura::Text warningText;
        bool hasWarning = false;
        Sakura::Div card;
        Sakura::Spacing spacing;
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
        this->config = std::move(config);

        title.update({
            .id = "RemoteTitle",
            .str = "Remote",
            .font = Sakura::Text::Font::Bold,
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

        activeCard.update({
            .id = "RemoteActiveCard",
            .padding = 16.0f,
            .rounding = 8.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
        });

        activeWarning.update({
            .id = "RemoteActiveWarning",
            .str = ICON_FA_TOWER_BROADCAST " Remote streaming is currently active.",
            .tone = Sakura::Text::Tone::Warning,
            .scale = 1.05f,
        });

        activeDescription.update({
            .id = "RemoteActiveDescription",
            .str = "Updates here take effect after the current session ends.",
            .tone = Sakura::Text::Tone::Disabled,
            .wrapped = true,
        });

        activeSpacing.update({
            .id = "RemoteActiveSpacing",
        });

        accessSection.update("RemoteAccessSection", "Access", "Who can join a remote session.");
        videoSection.update("RemoteVideoSection", "Video", "How the interface is encoded for viewers.");
        brokerSection.update("RemoteBrokerSection",
                             "Broker",
                             "The server on this machine that hosts the viewer page and connects viewers.",
                             "Do not expose the broker directly to the internet.");
        mediaSection.update("RemoteMediaSection", "Media", "Network settings for the video connection.");

        brokerAddressField.update({
            .id = "RemoteBrokerAddressField",
            .label = "Broker Address",
            .description = "The network address the broker listens on. Use 0.0.0.0 to accept viewers on every interface.",
        });

        brokerAddressInput.update({
            .id = "##app-settings-broker-address",
            .value = this->config.broker.address,
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.address = value.empty() ? Instance::Remote::Config::Broker{}.address : value;
                changeBroker(broker);
            },
        });

        brokerPortField.update({
            .id = "RemoteBrokerPortField",
            .label = "Broker Port",
            .description = "The port that serves the remote viewer page and its connections.",
        });

        brokerPortInput.update({
            .id = "##app-settings-broker-port",
            .value = jst::fmt::format("{}", this->config.broker.port),
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                U16 port = 0;
                if (portValue(value, port)) {
                    auto broker = this->config.broker;
                    broker.port = port;
                    changeBroker(broker);
                }
            },
        });

        brokerPublicUrlField.update({
            .id = "RemoteBrokerPublicUrlField",
            .label = "Broker Public URL",
            .description = "The address viewers use to reach this machine, written as scheme, host, and port only. Leave empty to detect the local network address.",
        });

        brokerPublicUrlInput.update({
            .id = "##app-settings-broker-public-url",
            .value = this->config.broker.publicUrl,
            .hint = "Automatic",
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                if (!value.empty() && !IsRemotePublicUrlValid(value)) {
                    return;
                }
                auto broker = this->config.broker;
                broker.publicUrl = value;
                changeBroker(broker);
            },
        });

        brokerCertificateField.update({
            .id = "RemoteBrokerCertificateField",
            .label = "Broker TLS Certificate",
            .description = "Path to a PEM certificate that enables HTTPS. Leave empty to serve plain HTTP.",
        });

        brokerCertificateInput.update({
            .id = "##app-settings-broker-certificate",
            .value = this->config.broker.certificate,
            .hint = "None",
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.certificate = value;
                changeBroker(broker);
            },
        });

        brokerCertificateBrowse.update({
            .id = "RemoteBrokerCertificateBrowse",
            .str = "Browse File",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                browseFile(this->config.broker.certificate, {"pem", "crt", "cer"}, [this](std::string path) {
                    auto broker = this->config.broker;
                    broker.certificate = std::move(path);
                    changeBroker(broker);
                });
            },
        });

        brokerKeyField.update({
            .id = "RemoteBrokerKeyField",
            .label = "Broker TLS Key",
            .description = "Path to the PEM private key that matches the certificate.",
            .divider = false,
        });

        brokerKeyInput.update({
            .id = "##app-settings-broker-key",
            .value = this->config.broker.key,
            .hint = "None",
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.key = value;
                changeBroker(broker);
            },
        });

        brokerKeyBrowse.update({
            .id = "RemoteBrokerKeyBrowse",
            .str = "Browse File",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                browseFile(this->config.broker.key, {"pem", "key"}, [this](std::string path) {
                    auto broker = this->config.broker;
                    broker.key = std::move(path);
                    changeBroker(broker);
                });
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
            .label = "Video Framerate",
            .description = "Target framerate for the outgoing video stream.",
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
            .label = "Video Encoder",
            .description = "The hardware or software encoder used to compress the video stream.",
            .divider = false,
        });

        encoderCombo.update({
            .id = "##app-settings-encoder",
            .options = encoderOptions(this->config.available),
            .value = encoderLabel(this->config.encoder, this->config.available),
            .disabled = this->config.available.empty(),
            .onChange = [this](const std::string& label) {
                if (this->config.onEncoderChange) {
                    this->config.onEncoderChange(encoderValue(label));
                }
            },
        });

        approvalField.update({
            .id = "RemoteApprovalField",
            .label = "Viewer Approval",
            .description = "Whether new viewers join automatically or wait for manual approval.",
            .divider = false,
        });

        approvalCombo.update({
            .id = "##app-settings-client-approval",
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

        transportField.update({
            .id = "RemoteTransportField",
            .label = "Media Transport",
            .description = "The network protocols offered for the media connection. TCP helps on networks that block UDP.",
        });

        transportCombo.update({
            .id = "##app-settings-transport",
            .options = transportOptions(),
            .value = GetRemoteTransportPrettyName(this->config.transport),
            .onChange = [this](const std::string& label) {
                if (this->config.onTransportChange) {
                    this->config.onTransportChange(transportValue(label));
                }
            },
        });

        portRangeField.update({
            .id = "RemotePortRangeField",
            .label = "Media Port Range",
            .description = "The local ports used for the media connection, written as MIN-MAX. Leave empty to let the system choose.",
        });

        portRangeInput.update({
            .id = "##app-settings-port-range",
            .value = GetRemotePortRangeName(this->config.minPort, this->config.maxPort),
            .hint = "Any",
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                U16 minPort = 0;
                U16 maxPort = 0;
                if (this->config.onPortRangeChange && ParseRemotePortRange(value, minPort, maxPort)) {
                    this->config.onPortRangeChange(minPort, maxPort);
                }
            },
        });

        mediaAddressField.update({
            .id = "RemoteMediaAddressField",
            .label = "Media Address",
            .description = "The only local IP address offered for the media connection. Use 127.0.0.1 when tunneling over SSH. Leave empty to offer every address.",
            .divider = false,
        });

        mediaAddressInput.update({
            .id = "##app-settings-media-address",
            .value = this->config.mediaAddress,
            .hint = "Automatic",
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                if (this->config.onMediaAddressChange) {
                    this->config.onMediaAddressChange(value);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        if (config.started) {
            activeCard.render(ctx, [&](const Sakura::Context& ctx) {
                activeWarning.render(ctx);
                activeSpacing.render(ctx);
                activeDescription.render(ctx);
            });
            activeSpacing.render(ctx);
        }

        accessSection.render(ctx, [&](const Sakura::Context& ctx) {
            approvalField.render(ctx, [&](const Sakura::Context& ctx) {
                approvalCombo.render(ctx);
            });
        });

        videoSection.render(ctx, [&](const Sakura::Context& ctx) {
            framerateField.render(ctx, [&](const Sakura::Context& ctx) {
                framerateCombo.render(ctx);
            });

            codecField.render(ctx, [&](const Sakura::Context& ctx) {
                codecCombo.render(ctx);
            });

            encoderField.render(ctx, [&](const Sakura::Context& ctx) {
                encoderCombo.render(ctx);
            });
        });

        brokerSection.render(ctx, [&](const Sakura::Context& ctx) {
            brokerAddressField.render(ctx, [&](const Sakura::Context& ctx) {
                brokerAddressInput.render(ctx);
            });

            brokerPortField.render(ctx, [&](const Sakura::Context& ctx) {
                brokerPortInput.render(ctx);
            });

            brokerPublicUrlField.render(ctx, [&](const Sakura::Context& ctx) {
                brokerPublicUrlInput.render(ctx);
            });

            brokerCertificateField.render(ctx, [&](const Sakura::Context& ctx) {
                brokerCertificateInput.render(ctx);
                brokerCertificateBrowse.render(ctx);
            });

            brokerKeyField.render(ctx, [&](const Sakura::Context& ctx) {
                brokerKeyInput.render(ctx);
                brokerKeyBrowse.render(ctx);
            });
        });

        mediaSection.render(ctx, [&](const Sakura::Context& ctx) {
            transportField.render(ctx, [&](const Sakura::Context& ctx) {
                transportCombo.render(ctx);
            });

            portRangeField.render(ctx, [&](const Sakura::Context& ctx) {
                portRangeInput.render(ctx);
            });

            mediaAddressField.render(ctx, [&](const Sakura::Context& ctx) {
                mediaAddressInput.render(ctx);
            });
        });
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
    Sakura::Div activeCard;
    Sakura::Text activeWarning;
    Sakura::Text activeDescription;
    Sakura::Spacing activeSpacing;
    Section accessSection;
    Section videoSection;
    Section brokerSection;
    Section mediaSection;
    Sakura::SettingField brokerAddressField;
    Sakura::SettingField brokerPortField;
    Sakura::SettingField brokerPublicUrlField;
    Sakura::SettingField brokerCertificateField;
    Sakura::SettingField brokerKeyField;
    Sakura::SettingField codecField;
    Sakura::SettingField framerateField;
    Sakura::SettingField encoderField;
    Sakura::SettingField approvalField;
    Sakura::SettingField transportField;
    Sakura::SettingField portRangeField;
    Sakura::SettingField mediaAddressField;
    Sakura::TextInput brokerAddressInput;
    Sakura::TextInput brokerPortInput;
    Sakura::TextInput brokerPublicUrlInput;
    Sakura::TextInput brokerCertificateInput;
    Sakura::TextInput brokerKeyInput;
    Sakura::TextInput portRangeInput;
    Sakura::TextInput mediaAddressInput;
    Sakura::Combo codecCombo;
    Sakura::Combo framerateCombo;
    Sakura::Combo encoderCombo;
    Sakura::Combo approvalCombo;
    Sakura::Button brokerCertificateBrowse;
    Sakura::Button brokerKeyBrowse;
    Sakura::Combo transportCombo;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REMOTE_HH
