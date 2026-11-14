#include "history.hh"

namespace Jetstream {

bool FlowgraphSnapshot::Block::sameRuntime(const Block& other) const {
    return type == other.type && device == other.device && runtime == other.runtime &&
           provider == other.provider;
}

bool FlowgraphSnapshot::Block::sameInput(const std::string& slot, const Block& other) const {
    const auto mine = inputs.find(slot);
    const auto theirs = other.inputs.find(slot);
    return mine != inputs.end() && theirs != other.inputs.end() &&
           mine->second.block == theirs->second.block && mine->second.port == theirs->second.port;
}

bool FlowgraphSnapshot::Block::sameInputs(const Block& other) const {
    return inputs.size() == other.inputs.size() &&
           std::all_of(inputs.begin(), inputs.end(), [&](const auto& entry) {
               return sameInput(entry.first, other);
           });
}

const FlowgraphSnapshot::Block* FlowgraphSnapshot::find(const std::string& name) const {
    const auto entry = std::find_if(blocks.begin(), blocks.end(), [&](const Block& block) {
        return block.name == name;
    });
    return entry == blocks.end() ? nullptr : &*entry;
}

bool FlowgraphHistory::canUndo() const {
    return cursor > 0;
}

bool FlowgraphHistory::canRedo() const {
    return cursor + 1 < entries.size();
}

bool FlowgraphHistory::unsaved() const {
    return untracked || (!entries.empty() && entries[cursor] != saved);
}

bool FlowgraphHistory::settled() const {
    return pendingEdits == 0 && !restoring && deferred.empty();
}

const FlowgraphSnapshot& FlowgraphHistory::current() const {
    return *entries[cursor];
}

void FlowgraphHistory::seed(Entry first) {
    saved = first;
    entries.push_back(std::move(first));
}

void FlowgraphHistory::markSaved(Entry exported, bool current) {
    if (current) {
        absorb(exported);
    }
    saved = std::move(exported);
    mergeKey.clear();
}

void FlowgraphHistory::commit(Entry next) {
    entries.resize(cursor + 1);
    entries.push_back(std::move(next));
    if (entries.size() > Capacity) {
        entries.erase(entries.begin());
    }
    cursor = entries.size() - 1;
}

void FlowgraphHistory::absorb(Entry next) {
    if (entries[cursor] == saved) {
        saved = next;
    }
    entries[cursor] = std::move(next);
}

void FlowgraphHistory::place(const std::string& block, const NodeMeta& node) {
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

bool FlowgraphHistory::settlePendingSteps() {
    const I64 last = static_cast<I64>(entries.size()) - 1;
    const U64 next = static_cast<U64>(std::clamp(static_cast<I64>(cursor) + pendingSteps, I64{0}, last));
    pendingSteps = 0;
    const bool moved = next != cursor;
    cursor = next;
    return moved;
}

}  // namespace Jetstream
