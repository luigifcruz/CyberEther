#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_TERMINAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_TERMINAL_HH

#include "layout.hh"
#include "workbench.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../model/tiles.hh"
#include "../support/terminal.hh"

#include "jetstream/platform.hh"

#include <algorithm>
#include <array>
#include <cctype>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace Jetstream {

struct TerminalActions {
    static constexpr I32 kMinZoom = -6;
    static constexpr I32 kMaxZoom = 24;

    static bool Openable(const std::string& uri);

    using Filter = std::tuple<MailOpenTerminal,
                              MailCloseTerminal,
                              MailCloseTerminalTab,
                              MailFocusTerminalTab,
                              MailSetTerminalTabLayout,
                              MailTerminalInput>;
    using Terminal = DefaultCompositorState::TerminalState::Item;

    struct Session {
        std::unique_ptr<TerminalSession> session;
        U64 revision = 0;
    };

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;
    std::unordered_map<U64, Session> sessions;
    U64 published = 0;

    TerminalActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks);
    Result handle(const MailOpenTerminal& msg);
    Result handle(const MailCloseTerminal& msg);
    Result handle(const MailCloseTerminalTab& msg);
    Result handle(const MailFocusTerminalTab& msg);
    Result handle(const MailSetTerminalTabLayout& msg);
    Result handle(const MailTerminalInput& msg);
    void reconcile();
    void shutdown();

 private:
    void publish(Terminal& terminal, Session& entry);
    TerminalSession::Theme palette() const;
    U64 open(const std::string& key);
    U64 open(TileMeta layout);
    void splice(TileMeta layout, const std::optional<U64>& tab,
                const std::function<void(const std::string&, const TileMeta&)>& replace);
    std::function<bool(const std::string&)> claimer(const std::optional<std::string>& flowgraph,
                                                    const std::optional<U64>& tab);
    void settle();
    std::string directory(const std::string& flowgraphId) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_TERMINAL_HH
