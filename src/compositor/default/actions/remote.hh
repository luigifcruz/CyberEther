#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_REMOTE_HH

#include "persist.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/instance_remote.hh"
#include "jetstream/settings.hh"

#include <tuple>

namespace Jetstream {

struct RemoteActions {
    using Filter = std::tuple<MailStartRemote,
                              MailStopRemote,
                              MailApproveRemoteClient,
                              MailRemoveRemoteClient,
                              MailSetRemoteBroker,
                              MailSetRemoteCodec,
                              MailSetRemoteFramerate,
                              MailSetRemoteEncoder,
                              MailSetRemoteAutoApprove,
                              MailSetRemoteTransport,
                              MailSetRemotePortRange,
                              MailSetRemoteMediaAddress>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    RemoteActions(DefaultCompositorState& state,
                  DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result handle(const MailStartRemote& msg) {
        const Result result = state.system.instance->remote()->create(msg.config);
        if (result == Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Success, 5000, "Remote streaming started.");
        } else {
            callbacks.notify(Sakura::ToastType::Error, 5000, "Failed to start remote streaming.");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailStopRemote&) {
        if (state.system.instance->remote()->destroy() == Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Info, 5000, "Remote streaming stopped.");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailApproveRemoteClient& msg) {
        if (state.system.instance->remote()->approveClient(msg.sessionId) == Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Success, 3000, "Client approved.");
        } else {
            callbacks.notify(Sakura::ToastType::Error, 3000, "Failed to approve client.");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailRemoveRemoteClient& msg) {
        if (state.system.instance->remote()->removeClient(msg.sessionId) == Result::SUCCESS) {
            callbacks.notify(Sakura::ToastType::Success, 3000, "Client removed.");
        } else {
            callbacks.notify(Sakura::ToastType::Error, 3000, "Failed to remove client.");
        }

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteBroker& msg) {
        state.remote.config.broker = msg.value;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.broker.address = msg.value.address;
            settings.remote.broker.port = msg.value.port;
            settings.remote.broker.publicUrl = msg.value.publicUrl;
            settings.remote.broker.certificate = msg.value.certificate;
            settings.remote.broker.key = msg.value.key;
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteCodec& msg) {
        state.remote.config.video.codec = msg.value;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.video.codec = GetRemoteCodecName(msg.value);
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteFramerate& msg) {
        state.remote.config.video.framerate = msg.value;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.video.framerate = msg.value;
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteEncoder& msg) {
        state.remote.config.video.encoder = msg.value;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.video.encoder = GetRemoteEncoderName(msg.value);
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteAutoApprove& msg) {
        state.remote.config.access.autoApprove = msg.value;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.access.autoApprove = msg.value;
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteTransport& msg) {
        state.remote.config.media.transport = msg.value;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.media.transport = GetRemoteTransportName(msg.value);
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemotePortRange& msg) {
        state.remote.config.media.minPort = msg.minPort;
        state.remote.config.media.maxPort = msg.maxPort;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.media.minPort = msg.minPort;
            settings.remote.media.maxPort = msg.maxPort;
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetRemoteMediaAddress& msg) {
        state.remote.config.media.address = msg.value;

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.remote.media.address = msg.value;
        }));

        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_REMOTE_HH
