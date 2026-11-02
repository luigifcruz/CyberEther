#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REMOTE_HH

#include "../components/callout.hh"
#include "../components/file_field.hh"
#include "../components/option_stepper.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include "jetstream/instance_remote.hh"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteSettingsView {
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
        Sakura::Callback<void(const Instance::Remote::Config::Broker&)> onBrokerChange;
        Sakura::Callback<void(const std::string&, std::vector<std::string>, Sakura::Callback<void(std::string)>)> onBrowseFile;
        Sakura::Callback<void(Instance::Remote::CodecType)> onCodecChange;
        Sakura::Callback<void(U32)> onFramerateChange;
        Sakura::Callback<void(Instance::Remote::EncoderType)> onEncoderChange;
        Sakura::Callback<void(bool)> onAutoApproveChange;
        Sakura::Callback<void(Instance::Remote::TransportType)> onTransportChange;
        Sakura::Callback<void(U16, U16)> onPortRangeChange;
        Sakura::Callback<void(const std::string&)> onMediaAddressChange;
    };

    void update(Config config) {
        this->config = std::move(config);

        header.update({
            .title = "Remote",
            .description = "Configure default remote streaming parameters.",
        });

        activeCallout.update({
            .str = "Changes apply when the current stream ends.",
            .tone = Callout::Tone::Warning,
            .font = Sakura::Text::Font::Sans,
            .rounding = Sakura::Rounding::Card,
        });
        activeTrailing.update({.lines = 4});

        approvalToggle.update({
            .value = this->config.autoApprove,
            .onChange = [this](bool value) {
                if (this->config.onAutoApproveChange) {
                    this->config.onAutoApproveChange(value);
                }
            },
        });

        framerateStepper.update({
            .options = {"15 fps", "30 fps", "60 fps", "120 fps"},
            .selected = framerateIndex(this->config.framerate),
            .fallback = 1,
            .onSelect = [this](U64 index) {
                if (this->config.onFramerateChange) {
                    this->config.onFramerateChange(framerates()[index]);
                }
            },
        });

        std::vector<Sakura::PillTabs::Item> codecItems;
        U64 codecIndex = 0;
        for (U64 index = 0; index < RemoteCodecTypes.size(); ++index) {
            codecItems.push_back({.label = codecLabel(RemoteCodecTypes[index])});
            if (RemoteCodecTypes[index] == this->config.codec) {
                codecIndex = index;
            }
        }
        codecTabs.update({
            .items = std::move(codecItems),
            .selected = codecIndex,
            .compact = true,
            .onSelect = [this](U64 index) {
                if (index < RemoteCodecTypes.size() && this->config.onCodecChange) {
                    this->config.onCodecChange(RemoteCodecTypes[index]);
                }
            },
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

        brokerAddressInput.update({
            .icon = ICON_LC_NETWORK,
            .value = this->config.broker.address,
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.address = value.empty() ? Instance::Remote::Config::Broker{}.address : value;
                changeBroker(broker);
            },
        });

        brokerPortInput.update({
            .icon = ICON_LC_ETHERNET_PORT,
            .value = jst::fmt::format("{}", this->config.broker.port),
            .onChange = [this](const std::string& value) {
                U16 port = 0;
                if (portValue(value, port)) {
                    auto broker = this->config.broker;
                    broker.port = port;
                    changeBroker(broker);
                }
            },
        });

        brokerPublicUrlInput.update({
            .icon = ICON_LC_GLOBE,
            .value = this->config.broker.publicUrl,
            .hint = "Automatic",
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
            .icon = ICON_LC_FILE_BADGE,
            .value = this->config.broker.certificate,
            .hint = "None",
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.certificate = value;
                changeBroker(broker);
            },
            .onBrowse = [this]() {
                browseFile(this->config.broker.certificate, {"pem", "crt", "cer"}, [this](std::string path) {
                    auto broker = this->config.broker;
                    broker.certificate = std::move(path);
                    changeBroker(broker);
                });
            },
        });

        brokerKeyField.update({
            .icon = ICON_LC_FILE_KEY,
            .value = this->config.broker.key,
            .hint = "None",
            .onChange = [this](const std::string& value) {
                auto broker = this->config.broker;
                broker.key = value;
                changeBroker(broker);
            },
            .onBrowse = [this]() {
                browseFile(this->config.broker.key, {"pem", "key"}, [this](std::string path) {
                    auto broker = this->config.broker;
                    broker.key = std::move(path);
                    changeBroker(broker);
                });
            },
        });

        std::vector<Sakura::PillTabs::Item> transportItems;
        U64 transportIndex = 0;
        for (U64 index = 0; index < RemoteTransportTypes.size(); ++index) {
            transportItems.push_back({.label = GetRemoteTransportPrettyName(RemoteTransportTypes[index])});
            if (RemoteTransportTypes[index] == this->config.transport) {
                transportIndex = index;
            }
        }
        transportTabs.update({
            .items = std::move(transportItems),
            .selected = transportIndex,
            .compact = true,
            .onSelect = [this](U64 index) {
                if (index < RemoteTransportTypes.size() && this->config.onTransportChange) {
                    this->config.onTransportChange(RemoteTransportTypes[index]);
                }
            },
        });

        portRangeInput.update({
            .icon = ICON_LC_ETHERNET_PORT,
            .value = GetRemotePortRangeName(this->config.minPort, this->config.maxPort),
            .hint = "Any",
            .onChange = [this](const std::string& value) {
                U16 minPort = 0;
                U16 maxPort = 0;
                if (this->config.onPortRangeChange && ParseRemotePortRange(value, minPort, maxPort)) {
                    this->config.onPortRangeChange(minPort, maxPort);
                }
            },
        });

        mediaAddressInput.update({
            .icon = ICON_LC_NETWORK,
            .value = this->config.mediaAddress,
            .hint = "Automatic",
            .onChange = [this](const std::string& value) {
                if (this->config.onMediaAddressChange) {
                    this->config.onMediaAddressChange(value);
                }
            },
        });

        accessGroup.update({
            .title = "Access",
            .description = "Who can join a remote session.",
            .rows = {{
                .label = "Auto Approve Viewers",
                .description = "Let new viewers join without waiting for manual approval.",
                .control = &approvalToggle,
            }},
        });
        videoGroup.update({
            .title = "Video",
            .description = "How the interface is encoded for viewers.",
            .rows = {
                {
                    .label = "Video Framerate",
                    .description = "Target framerate for the outgoing video stream.",
                    .control = &framerateStepper,
                },
                {
                    .label = "Video Codec",
                    .description = "The video compression format used for streaming.",
                    .control = &codecTabs,
                    .wide = true,
                },
                {
                    .label = "Video Encoder",
                    .description = "The hardware or software encoder used to compress the video stream.",
                    .control = &encoderCombo,
                },
            },
        });
        brokerGroup.update({
            .title = "Broker",
            .description = "The server on this machine that hosts the viewer page and connects viewers.",
            .warning = "Do not expose the broker directly to the internet.",
            .rows = {
                {
                    .label = "Broker Address",
                    .description = "The network address the broker listens on. Use 0.0.0.0 to accept viewers on every interface.",
                    .control = &brokerAddressInput,
                },
                {
                    .label = "Broker Port",
                    .description = "The port that serves the remote viewer page and its connections.",
                    .control = &brokerPortInput,
                },
                {
                    .label = "Broker Public URL",
                    .description = "The address viewers use to reach this machine, written as scheme, host, and port only. Leave empty to detect the local network address.",
                    .control = &brokerPublicUrlInput,
                },
                {
                    .label = "Broker TLS Certificate",
                    .description = "Path to a PEM certificate that enables HTTPS. Leave empty to serve plain HTTP.",
                    .control = &brokerCertificateField,
                },
                {
                    .label = "Broker TLS Key",
                    .description = "Path to the PEM private key that matches the certificate.",
                    .control = &brokerKeyField,
                },
            },
        });
        mediaGroup.update({
            .title = "Media",
            .description = "Network settings for the video connection.",
            .rows = {
                {
                    .label = "Media Transport",
                    .description = "The network protocols offered for the media connection. TCP helps on networks that block UDP.",
                    .control = &transportTabs,
                    .wide = true,
                },
                {
                    .label = "Media Port Range",
                    .description = "The local ports used for the media connection, written as MIN-MAX. Leave empty to let the system choose.",
                    .control = &portRangeInput,
                },
                {
                    .label = "Media Address",
                    .description = "The only local IP address offered for the media connection. Use 127.0.0.1 when tunneling over SSH. Leave empty to offer every address.",
                    .control = &mediaAddressInput,
                },
            },
        });

        std::vector<Sakura::Stack::Item> items = {
            {.child = &header},
        };
        if (this->config.started) {
            items.push_back({.child = &activeCallout});
            items.push_back({.child = &activeTrailing});
        }
        items.push_back({.child = &accessGroup});
        items.push_back({.child = &videoGroup});
        items.push_back({.child = &brokerGroup});
        items.push_back({.child = &mediaGroup});
        stack.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = std::move(items),
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    void browseFile(const std::string& path,
                    std::vector<std::string> extensions,
                    Sakura::Callback<void(std::string)> onSelect) const {
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

    static const std::vector<U32>& framerates() {
        static const std::vector<U32> values = {15, 30, 60, 120};
        return values;
    }

    static U64 framerateIndex(U32 framerate) {
        const auto& values = framerates();
        const auto it = std::find(values.begin(), values.end(), framerate);
        return it == values.end() ? 1 : static_cast<U64>(it - values.begin());
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

    static Instance::Remote::EncoderType encoderValue(const std::string& label) {
        try {
            return StringToRemoteEncoder(label);
        } catch (const Result&) {
            return Instance::Remote::EncoderType::Auto;
        }
    }

    Config config;
    SettingsHeader header;
    Callout activeCallout;
    Sakura::Spacer activeTrailing;
    Sakura::Toggle approvalToggle;
    OptionStepper framerateStepper;
    Sakura::PillTabs codecTabs;
    Sakura::Combo encoderCombo;
    Sakura::PillInput brokerAddressInput;
    Sakura::PillInput brokerPortInput;
    Sakura::PillInput brokerPublicUrlInput;
    FileField brokerCertificateField;
    FileField brokerKeyField;
    Sakura::PillTabs transportTabs;
    Sakura::PillInput portRangeInput;
    Sakura::PillInput mediaAddressInput;
    SectionGroup accessGroup;
    SectionGroup videoGroup;
    SectionGroup brokerGroup;
    SectionGroup mediaGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REMOTE_HH
