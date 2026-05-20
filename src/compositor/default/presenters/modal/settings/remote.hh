#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REMOTE_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/remote.hh"

namespace Jetstream {

struct RemoteSettingsPresenter {
    const PresenterContext& context;

    explicit RemoteSettingsPresenter(const PresenterContext& context) : context(context) {}

    RemoteSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto remote = context.state.system.instance->remote();
        const auto available = remote->available(context.state.remote.codec);

        return RemoteSettingsPanel::Config{
            .started = context.state.remote.started,
            .brokerUrl = context.state.remote.brokerUrl,
            .codec = context.state.remote.codec,
            .framerate = context.state.remote.framerate,
            .encoder = context.state.remote.encoder,
            .available = available,
            .autoJoinSessions = context.state.remote.autoJoinSessions,
            .onBrokerUrlChange = [enqueue](const std::string& value) {
                enqueue(MailSetRemoteBrokerUrl{.value = value});
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
            .onAutoJoinSessionsChange = [enqueue](bool value) {
                enqueue(MailSetRemoteAutoJoinSessions{.value = value});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_REMOTE_HH
