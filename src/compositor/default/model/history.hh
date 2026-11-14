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

        bool sameRuntime(const Block& other) const;
        bool sameInput(const std::string& slot, const Block& other) const;
        bool sameInputs(const Block& other) const;
    };

    std::vector<Block> blocks;

    const Block* find(const std::string& name) const;
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

    bool canUndo() const;
    bool canRedo() const;
    bool unsaved() const;
    bool settled() const;
    const FlowgraphSnapshot& current() const;
    void seed(Entry first);
    void markSaved(Entry exported, bool current);
    void commit(Entry next);
    void absorb(Entry next);
    void place(const std::string& block, const NodeMeta& node);
    bool settlePendingSteps();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_HISTORY_HH
