#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LIVE_MARKDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LIVE_MARKDOWN_HH

#include "../live_markdown.hh"
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

    explicit LiveMarkdownActions(FlowgraphState& state) : state(state) {}

    void refresh() {
        const auto now = Clock::now();
        std::unordered_set<std::string> seen;
        std::unordered_set<std::string> live;

        for (const auto& [flowgraphId, flowgraph] : state.items) {
            const auto blocks = state.blocks.find(flowgraphId);
            if (!flowgraph || blocks == state.blocks.end()) {
                continue;
            }
            const auto& environment = flowgraph->environment();
            const U64 epoch = environment.epoch();
            for (const auto& block : blocks->second) {
                for (const auto& entry : block.data.interfaceConfigs) {
                    if (Parser::Get<std::string>(entry.format, "type") != "markdown") {
                        continue;
                    }
                    std::string source;
                    if (Parser::Deserialize(block.data.config, entry.name, source) != Result::SUCCESS) {
                        continue;
                    }

                    const auto key = FlowgraphState::LiveMarkdownKey(flowgraphId, block.name, entry.name);
                    auto& cached = entries[key];
                    seen.insert(key);
                    const bool edited = cached.compiled.source() != source;
                    if (edited) {
                        cached.compiled = LiveMarkdown(std::move(source));
                    }
                    if (!cached.compiled.dynamic()) {
                        continue;
                    }

                    live.insert(key);
                    const bool stale = cached.epoch != epoch && now - cached.refreshed >= RefreshInterval;
                    if (edited || stale || !state.liveMarkdown.contains(key)) {
                        state.liveMarkdown[key] = cached.compiled.expand(environment);
                        cached.epoch = epoch;
                        cached.refreshed = now;
                    }
                }
            }
        }

        std::erase_if(state.liveMarkdown, [&](const auto& item) { return !live.contains(item.first); });
        std::erase_if(entries, [&](const auto& item) { return !seen.contains(item.first); });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LIVE_MARKDOWN_HH
