#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_REMOTE_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../views/modal/remote.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct DefaultRemotePresenter {
    using ModalContent = DefaultCompositorState::ModalState::Content;
    using SettingsSection = DefaultCompositorState::SettingsState::Section;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    DefaultRemotePresenter(DefaultCompositorState& state,
                           DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    RemoteView::Config build() const {
        const auto enqueue = callbacks.enqueueMail;
        const bool isStarted = state.remote.started;
        const std::string remoteBrokerUrl = state.remote.brokerUrl;
        const auto remoteCodec = state.remote.codec;
        const auto remoteEncoder = state.remote.encoder;
        const bool remoteAutoJoinSessions = state.remote.autoJoinSessions;
        const U32 remoteFramerate = state.remote.framerate;

        return RemoteView::Config{
            .started = isStarted,
            .inviteUrl = isStarted ? state.remote.inviteUrl : "",
            .roomId = isStarted ? state.remote.roomId : "",
            .accessToken = isStarted ? state.remote.accessToken : "",
            .clients = isStarted ? state.remote.clients : std::vector<Instance::Remote::ClientInfo>{},
            .waitlist = isStarted ? state.remote.waitlist : std::vector<std::string>{},
            .onStart = [enqueue,
                        remoteBrokerUrl,
                        remoteCodec,
                        remoteEncoder,
                        remoteAutoJoinSessions,
                        remoteFramerate]() {
                Instance::Remote::Config remoteConfig;
                remoteConfig.broker = remoteBrokerUrl;
                remoteConfig.codec = remoteCodec;
                remoteConfig.encoder = remoteEncoder;
                remoteConfig.autoJoinSessions = remoteAutoJoinSessions;
                remoteConfig.framerate = remoteFramerate;
                enqueue(MailStartRemote{remoteConfig});
            },
            .onConfigure = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::Remote});
            },
            .onStop = [enqueue]() {
                enqueue(MailStopRemote{});
            },
            .onApprove = [enqueue](const std::string& code) {
                enqueue(MailApproveRemoteClient{code});
            },
            .onOpenUrl = [enqueue](const std::string& url) {
                enqueue(MailOpenUrl{.url = url, .notifyResult = true});
            },
            .onCopy = [enqueue](const std::string& label, const std::string& value) {
                enqueue(MailCopyText{.label = label, .value = value});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_REMOTE_HH
