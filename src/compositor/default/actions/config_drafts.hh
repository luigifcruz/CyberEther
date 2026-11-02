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

    explicit ConfigDraftActions(FlowgraphState& state) : state(state) {}

    Result handle(const MailSetConfigDraft& msg) {
        const auto key = FlowgraphState::FieldKey(msg.flowgraph, msg.block, msg.field);
        diverged.erase(key);
        if (!msg.text.has_value() || *msg.text == msg.base) {
            state.configDrafts.erase(key);
            return Result::SUCCESS;
        }
        state.configDrafts[key] = {
            .flowgraph = msg.flowgraph,
            .block = msg.block,
            .field = msg.field,
            .base = msg.base,
            .text = *msg.text,
        };
        return Result::SUCCESS;
    }

    void refresh() {
        refresh(Clock::now());
    }

    void refresh(const Clock::time_point now) {
        enum class Status { Current, Diverged, Closed };

        const auto status = [&](const FlowgraphState::ConfigDraft& draft) {
            const auto blocks = state.blocks.find(draft.flowgraph);
            if (blocks == state.blocks.end()) {
                return Status::Closed;
            }
            const auto block = std::find_if(blocks->second.begin(), blocks->second.end(), [&](const auto& snapshot) {
                return snapshot.name == draft.block;
            });
            std::string committed;
            if (block == blocks->second.end() || !block->data.config.contains(draft.field) ||
                Parser::Deserialize(block->data.config, draft.field, committed) != Result::SUCCESS) {
                return Status::Diverged;
            }
            return committed == draft.base ? Status::Current : Status::Diverged;
        };

        std::erase_if(state.configDrafts, [&](const auto& item) {
            const auto& [key, draft] = item;
            switch (status(draft)) {
                case Status::Current:
                    diverged.erase(key);
                    return false;
                case Status::Diverged: {
                    const auto [since, inserted] = diverged.try_emplace(key, now);
                    if (inserted || now - since->second < SettleInterval) {
                        return false;
                    }
                    diverged.erase(since);
                    return true;
                }
                case Status::Closed:
                    diverged.erase(key);
                    return true;
            }
            return false;
        });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CONFIG_DRAFTS_HH
