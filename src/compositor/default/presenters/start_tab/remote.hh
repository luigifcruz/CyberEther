#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_REMOTE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/start_tab/remote.hh"

#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace Jetstream {

struct RemoteSectionPresenter {
    const PresenterContext& context;

    explicit RemoteSectionPresenter(const PresenterContext& context) : context(context) {}

    std::optional<RemoteSectionView::Config> build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& remote = context.state.remote;
        if (!remote.supported) {
            return std::nullopt;
        }
        RemoteSectionView::Config config{
            .started = remote.started,
            .onStart = [enqueue, settings = remote.config]() {
                enqueue(MailStartRemote{settings});
            },
            .onStop = [enqueue]() {
                enqueue(MailStopRemote{});
            },
            .onSettings = [enqueue]() {
                enqueue(MailOpenSettings{.section = SettingsSection::Remote});
            },
            .onCopyLink = [enqueue, url = remote.inviteUrl]() {
                enqueue(MailCopyText{.label = "Invite link", .value = url});
            },
            .onShowQr = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::RemoteQr});
            },
            .onApprove = [enqueue](const std::string& sessionId) {
                enqueue(MailApproveRemoteClient{sessionId});
            },
            .onRemove = [enqueue](const std::string& sessionId) {
                enqueue(MailRemoveRemoteClient{sessionId});
            },
        };
        if (!remote.started) {
            return config;
        }
        const std::set<std::string> waiting(remote.waitlist.begin(), remote.waitlist.end());
        std::set<std::string> seen;
        for (const auto& client : remote.clients) {
            if (client.sessionId.empty() || !seen.insert(client.sessionId).second) {
                continue;
            }
            config.viewers.push_back({
                .id = client.sessionId,
                .code = Code(client.sessionId),
                .waiting = waiting.contains(client.sessionId),
            });
        }
        for (const auto& sessionId : remote.waitlist) {
            if (sessionId.empty() || !seen.insert(sessionId).second) {
                continue;
            }
            config.viewers.push_back({.id = sessionId, .code = Code(sessionId), .waiting = true});
        }
        return config;
    }

 private:
    static std::string Code(const std::string& sessionId) {
        std::string code = sessionId.size() <= 6 ? sessionId : sessionId.substr(sessionId.size() - 6);
        std::transform(code.begin(), code.end(), code.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return code;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_REMOTE_HH
