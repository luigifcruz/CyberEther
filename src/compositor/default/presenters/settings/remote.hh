#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REMOTE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/remote.hh"

namespace Jetstream {

struct RemoteSettingsPresenter {
    const PresenterContext& context;

    explicit RemoteSettingsPresenter(const PresenterContext& context) : context(context) {}

    RemoteSettingsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto remote = context.state.system.instance->remote();
        const auto available = remote->available(context.state.remote.config.video.codec);

        return RemoteSettingsView::Config{
            .started = context.state.remote.started,
            .broker = context.state.remote.config.broker,
            .codec = context.state.remote.config.video.codec,
            .framerate = context.state.remote.config.video.framerate,
            .encoder = context.state.remote.config.video.encoder,
            .available = available,
            .autoApprove = context.state.remote.config.access.autoApprove,
            .transport = context.state.remote.config.media.transport,
            .minPort = context.state.remote.config.media.minPort,
            .maxPort = context.state.remote.config.media.maxPort,
            .mediaAddress = context.state.remote.config.media.address,
            .onBrokerChange = [enqueue](const Instance::Remote::Config::Broker& value) {
                enqueue(MailSetRemoteBroker{.value = value});
            },
            .onBrowseFile = [enqueue](const std::string& currentPath,
                                      std::vector<std::string> extensions,
                                      std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = currentPath,
                    .save = false,
                    .extensions = std::move(extensions),
                    .onSelect = std::move(onSelect),
                });
            },
            .onCodecChange = [enqueue](Instance::Remote::CodecType value) {
                enqueue(MailSetRemoteCodec{.value = value});
            },
            .onFramerateChange = [enqueue](U32 value) {
                enqueue(MailSetRemoteFramerate{.value = value});
            },
            .onEncoderChange = [enqueue](Instance::Remote::EncoderType value) {
                enqueue(MailSetRemoteEncoder{.value = value});
            },
            .onAutoApproveChange = [enqueue](bool value) {
                enqueue(MailSetRemoteAutoApprove{.value = value});
            },
            .onTransportChange = [enqueue](Instance::Remote::TransportType value) {
                enqueue(MailSetRemoteTransport{.value = value});
            },
            .onPortRangeChange = [enqueue](U16 minPort, U16 maxPort) {
                enqueue(MailSetRemotePortRange{.minPort = minPort, .maxPort = maxPort});
            },
            .onMediaAddressChange = [enqueue](const std::string& value) {
                enqueue(MailSetRemoteMediaAddress{.value = value});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REMOTE_HH
