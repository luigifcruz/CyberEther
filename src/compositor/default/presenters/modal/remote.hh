#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/remote.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct RemoteStreamingModalPresenter {
    const PresenterContext& context;

    explicit RemoteStreamingModalPresenter(const PresenterContext& context) : context(context) {}

    RemoteView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const bool isStarted = context.state.remote.started;
        const auto remoteConfig = context.state.remote.config;

        return RemoteView::Config{
            .started = isStarted,
            .inviteUrl = isStarted ? context.state.remote.inviteUrl : "",
            .roomId = isStarted ? context.state.remote.roomId : "",
            .accessToken = isStarted ? context.state.remote.accessToken : "",
            .clients = isStarted ? context.state.remote.clients : std::vector<Instance::Remote::ClientInfo>{},
            .waitlist = isStarted ? context.state.remote.waitlist : std::vector<std::string>{},
            .onStart = [enqueue, remoteConfig]() {
                enqueue(MailStartRemote{remoteConfig});
            },
            .onConfigure = [enqueue]() {
                enqueue(MailCloseModal{});
                enqueue(MailOpenSettings{.section = SettingsSection::Remote});
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

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_HH
