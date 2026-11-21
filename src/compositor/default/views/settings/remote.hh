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

    void update(Config config);
    Sakura::Component& content();

 private:
    void browseFile(const std::string& path,
                    std::vector<std::string> extensions,
                    Sakura::Callback<void(std::string)> onSelect) const;
    void changeBroker(const Instance::Remote::Config::Broker& broker) const;
    static bool portValue(const std::string& label, U16& port);
    static std::string codecLabel(Instance::Remote::CodecType codec);
    static const std::vector<U32>& framerates();
    static U64 framerateIndex(U32 framerate);
    static std::string encoderLabel(Instance::Remote::EncoderType encoder,
                                    const std::vector<Instance::Remote::EncoderType>& encoders);
    static std::vector<std::string> encoderOptions(const std::vector<Instance::Remote::EncoderType>& encoders);
    static Instance::Remote::EncoderType encoderValue(const std::string& label);

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
