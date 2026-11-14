#include "live_markdown.hh"

namespace Jetstream {

LiveMarkdownActions::LiveMarkdownActions(FlowgraphState& state) : state(state) {}

void LiveMarkdownActions::refresh() {
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

                const auto key = FlowgraphState::FieldKey(flowgraphId, block.name, entry.name);
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

}  // namespace Jetstream
