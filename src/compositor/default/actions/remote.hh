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
                  DefaultCompositorCallbacks& callbacks);
    Result handle(const MailStartRemote& msg);
    Result handle(const MailStopRemote&);
    Result handle(const MailApproveRemoteClient& msg);
    Result handle(const MailRemoveRemoteClient& msg);
    Result handle(const MailSetRemoteBroker& msg);
    Result handle(const MailSetRemoteCodec& msg);
    Result handle(const MailSetRemoteFramerate& msg);
    Result handle(const MailSetRemoteEncoder& msg);
    Result handle(const MailSetRemoteAutoApprove& msg);
    Result handle(const MailSetRemoteTransport& msg);
    Result handle(const MailSetRemotePortRange& msg);
    Result handle(const MailSetRemoteMediaAddress& msg);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_REMOTE_HH
