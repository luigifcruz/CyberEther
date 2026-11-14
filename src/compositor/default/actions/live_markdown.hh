#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LIVE_MARKDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LIVE_MARKDOWN_HH

#include "../support/live_markdown.hh"
#include "../model/state.hh"

#include "jetstream/flowgraph_environment.hh"
#include "jetstream/parser.hh"

#include <chrono>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Jetstream {

struct LiveMarkdownActions {
    using FlowgraphState = DefaultCompositorState::FlowgraphState;
    using Clock = std::chrono::steady_clock;

    static constexpr auto RefreshInterval = std::chrono::milliseconds(100);

    struct Entry {
        LiveMarkdown compiled;
        U64 epoch = 0;
        Clock::time_point refreshed;
    };

    FlowgraphState& state;
    std::unordered_map<std::string, Entry> entries;

    explicit LiveMarkdownActions(FlowgraphState& state);
    void refresh();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LIVE_MARKDOWN_HH
