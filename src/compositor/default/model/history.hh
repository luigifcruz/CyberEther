#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_HISTORY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_HISTORY_HH

#include "meta.hh"

#include "jetstream/parser.hh"
#include "jetstream/render/sakura/callback.hh"
#include "jetstream/tensor_link.hh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Jetstream {

struct FlowgraphSnapshot {
    struct Block {
        std::string name;
        std::string type;
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider;
        bool frozen = false;
        Parser::Map config;
        std::size_t configHash = 0;
        std::map<std::string, BlockEndpoint> inputs;
        Parser::Map metadata;
        std::optional<NodeMeta> node;

        bool sameRuntime(const Block& other) const {
            return type == other.type && device == other.device && runtime == other.runtime &&
                   provider == other.provider;
        }

        bool sameInput(const std::string& slot, const Block& other) const {
            const auto mine = inputs.find(slot);
            const auto theirs = other.inputs.find(slot);
            return mine != inputs.end() && theirs != other.inputs.end() &&
                   mine->second.block == theirs->second.block && mine->second.port == theirs->second.port;
        }

        bool sameInputs(const Block& other) const {
            return inputs.size() == other.inputs.size() &&
                   std::all_of(inputs.begin(), inputs.end(), [&](const auto& entry) {
                       return sameInput(entry.first, other);
                   });
        }
    };

    std::vector<Block> blocks;

    const Block* find(const std::string& name) const {
        const auto entry = std::find_if(blocks.begin(), blocks.end(), [&](const Block& block) {
            return block.name == name;
        });
        return entry == blocks.end() ? nullptr : &*entry;
    }
};

struct FlowgraphEdit {
    using Clock = std::chrono::steady_clock;

    std::string flowgraph;
    std::string mergeKey;
    Sakura::Callback<Result()> command;
    bool silent = false;
    bool layout = false;
    Sakura::Callback<void(Result)> onComplete;
    Clock::time_point requestedAt = Clock::now();
};

struct FlowgraphHistory {
    using Clock = FlowgraphEdit::Clock;
    using Entry = std::shared_ptr<const FlowgraphSnapshot>;

    static constexpr U64 Capacity = 128;

    std::vector<Entry> entries;
    U64 cursor = 0;
    Entry saved;
    bool untracked = false;
    U64 moduleConfigEdits = 0;

    U64 pendingEdits = 0;
    I64 pendingSteps = 0;
    bool restoring = false;
    std::vector<std::variant<FlowgraphEdit, I64>> deferred;

    std::string mergeKey;
    Clock::time_point mergedAt;

    bool canUndo() const {
        return cursor > 0;
    }

    bool canRedo() const {
        return cursor + 1 < entries.size();
    }

    bool unsaved() const {
        return untracked || (!entries.empty() && entries[cursor] != saved);
    }

    bool settled() const {
        return pendingEdits == 0 && !restoring && deferred.empty();
    }

    const FlowgraphSnapshot& current() const {
        return *entries[cursor];
    }

    void seed(Entry first) {
        saved = first;
        entries.push_back(std::move(first));
    }

    void markSaved(Entry exported, bool current) {
        if (current) {
            absorb(exported);
        }
        saved = std::move(exported);
        mergeKey.clear();
    }

    void commit(Entry next) {
        entries.resize(cursor + 1);
        entries.push_back(std::move(next));
        if (entries.size() > Capacity) {
            entries.erase(entries.begin());
        }
        cursor = entries.size() - 1;
    }

    void absorb(Entry next) {
        if (entries[cursor] == saved) {
            saved = next;
        }
        entries[cursor] = std::move(next);
    }

    void place(const std::string& block, const NodeMeta& node) {
        if (entries.empty()) {
            return;
        }
        const auto* existing = current().find(block);
        if (!existing || existing->node.has_value()) {
            return;
        }
        auto next = std::make_shared<FlowgraphSnapshot>(current());
        for (auto& entry : next->blocks) {
            if (entry.name == block) {
                entry.node = node;
            }
        }
        absorb(std::move(next));
    }

    bool settlePendingSteps() {
        const I64 last = static_cast<I64>(entries.size()) - 1;
        const U64 next = static_cast<U64>(std::clamp(static_cast<I64>(cursor) + pendingSteps, I64{0}, last));
        pendingSteps = 0;
        const bool moved = next != cursor;
        cursor = next;
        return moved;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_HISTORY_HH
