#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_REMOTE_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/render/sakura/sakura.hh"

#include <tuple>

namespace Jetstream {

struct RemoteActions {
    using Filter = std::tuple<MailStartRemote,
                              MailStopRemote,
                              MailApproveRemoteClient,
                              MailSetRemoteBrokerUrl,
                              MailSetRemoteCodec,
                              MailSetRemoteFramerate,
                              MailSetRemoteEncoder,
                              MailSetRemoteAutoJoinSessions>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    Result handle(const MailStartRemote& msg) {
        const Result result = state.system.instance->remote()->create(msg.config);
        if (result == Result::SUCCESS) {
            Sakura::Notify(Sakura::NotificationType::Success, 5000, "Remote streaming started.");
        } else {
            Sakura::Notify(Sakura::NotificationType::Error, 5000, "Failed to start remote streaming.");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailStopRemote&) {
        if (state.system.instance->remote()->destroy() == Result::SUCCESS) {
            Sakura::Notify(Sakura::NotificationType::Info, 5000, "Remote streaming stopped.");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailApproveRemoteClient& msg) {
        if (state.system.instance->remote()->approveClient(msg.code) == Result::SUCCESS) {
            Sakura::Notify(Sakura::NotificationType::Success, 3000, "Client approved.");
        } else {
            Sakura::Notify(Sakura::NotificationType::Error, 3000, "Failed to approve client.");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteBrokerUrl& msg) {
        state.remote.brokerUrl = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteCodec& msg) {
        state.remote.codec = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteFramerate& msg) {
        state.remote.framerate = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteEncoder& msg) {
        state.remote.encoder = msg.value;
        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteAutoJoinSessions& msg) {
        state.remote.autoJoinSessions = msg.value;
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_REMOTE_HH
