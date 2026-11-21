#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CONFIG_DRAFTS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CONFIG_DRAFTS_HH

#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/parser.hh"

#include <algorithm>
#include <chrono>
#include <string>
#include <tuple>
#include <unordered_map>

namespace Jetstream {

struct ConfigDraftActions {
    using FlowgraphState = DefaultCompositorState::FlowgraphState;
    using Clock = std::chrono::steady_clock;
    using Filter = std::tuple<MailSetConfigDraft>;

    static constexpr auto SettleInterval = std::chrono::seconds(2);

    FlowgraphState& state;
    std::unordered_map<std::string, Clock::time_point> diverged;

    explicit ConfigDraftActions(FlowgraphState& state);
    Result handle(const MailSetConfigDraft& msg);
    void refresh();
    void refresh(const Clock::time_point now);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CONFIG_DRAFTS_HH
