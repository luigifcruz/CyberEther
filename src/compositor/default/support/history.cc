#include "history.hh"

#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/parser.hh"
#include "jetstream/tensor_link.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace Jetstream {

namespace {

void Combine(std::size_t& seed, std::size_t next) {
    seed ^= next + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

bool Near(F32 a, F32 b) {
    return std::abs(a - b) <= 0.5f;
}

bool Applied(Result result) {
    return result == Result::SUCCESS || result == Result::RELOAD;
}

bool Failed(Result result) {
    return result != Result::SUCCESS && result != Result::INCOMPLETE && result != Result::RELOAD;
}

std::vector<const FlowgraphSnapshot::Block*> DependencyOrder(const FlowgraphSnapshot& snapshot) {
    std::vector<const FlowgraphSnapshot::Block*> order;
    std::unordered_set<std::string> placed;
    order.reserve(snapshot.blocks.size());
    while (order.size() < snapshot.blocks.size()) {
        const auto before = order.size();
        for (const auto& block : snapshot.blocks) {
            const bool ready = std::all_of(block.inputs.begin(), block.inputs.end(), [&](const auto& input) {
                return placed.contains(input.second.block) || !snapshot.find(input.second.block);
            });
            if (!placed.contains(block.name) && ready) {
                placed.insert(block.name);
                order.push_back(&block);
            }
        }
        if (order.size() == before) {
            break;
        }
    }
    for (const auto& block : snapshot.blocks) {
        if (!placed.contains(block.name)) {
            order.push_back(&block);
        }
    }
    return order;
}

Result Capture(const Flowgraph& flowgraph, FlowgraphSnapshot& snapshot) {
    std::vector<std::string> names;
    JST_CHECK(flowgraph.view().keys(names));

    snapshot.blocks.clear();
    snapshot.blocks.reserve(names.size());
    for (const auto& name : names) {
        Flowgraph::View::BlockData data;
        JST_CHECK(flowgraph.view().block(name, data));

        FlowgraphSnapshot::Block block{
            .name = name,
            .type = data.type,
            .device = data.device,
            .runtime = data.runtime,
            .provider = data.provider,
            .frozen = data.frozen,
            .config = std::move(data.config),
        };
        block.configHash = FlowgraphValueFingerprint(std::any(block.config));
        for (const auto& [slot, link] : data.inputs) {
            if (link.external.has_value()) {
                block.inputs[slot] = *link.external;
            }
        }

        std::vector<std::string> keys;
        JST_CHECK(flowgraph.metadata().keys(keys, name));
        for (const auto& key : keys) {
            Parser::Map value;
            JST_CHECK(flowgraph.metadata().get(key, value, name));
            block.metadata[key] = std::move(value);
        }

        NodeMeta node;
        if (flowgraph.metadata().tryGet("node", node, name)) {
            block.node = node;
        }
        snapshot.blocks.push_back(std::move(block));
    }
    return Result::SUCCESS;
}

}  // namespace

std::size_t FlowgraphValueFingerprint(const std::any& value) {
    if (!value.has_value()) {
        return 0;
    }

    std::size_t seed = value.type().hash_code();
    if (const auto* map = std::any_cast<Parser::Map>(&value)) {
        std::size_t entries = 0;
        for (const auto& [key, entry] : *map) {
            std::size_t pair = std::hash<std::string>{}(key);
            Combine(pair, FlowgraphValueFingerprint(entry));
            entries += pair;
        }
        Combine(seed, entries);
    } else if (const auto* sequence = std::any_cast<Parser::Sequence>(&value)) {
        for (const auto& entry : *sequence) {
            Combine(seed, FlowgraphValueFingerprint(entry));
        }
    } else if (std::string encoded; Parser::TryTypedToString(value, encoded)) {
        Combine(seed, std::hash<std::string>{}(encoded));
    }
    return seed;
}

FlowgraphHistory::Entry CaptureFlowgraphSnapshot(const Flowgraph& flowgraph) {
    auto snapshot = std::make_shared<FlowgraphSnapshot>();
    if (Capture(flowgraph, *snapshot) != Result::SUCCESS) {
        return nullptr;
    }
    return snapshot;
}

FlowgraphChange ClassifyFlowgraphChange(const FlowgraphSnapshot& previous, const FlowgraphSnapshot& next) {
    if (previous.blocks.size() != next.blocks.size()) {
        return FlowgraphChange::Content;
    }

    FlowgraphChange change = FlowgraphChange::None;
    for (const auto& block : next.blocks) {
        const auto* before = previous.find(block.name);
        if (!before || !block.sameRuntime(*before) || block.frozen != before->frozen ||
            block.configHash != before->configHash || !block.sameInputs(*before)) {
            return FlowgraphChange::Content;
        }
        if (block.node.has_value() && before->node.has_value() &&
            (!Near(block.node->x, before->node->x) || !Near(block.node->y, before->node->y) ||
             !Near(block.node->width, before->node->width) || !Near(block.node->height, before->node->height))) {
            change = FlowgraphChange::Layout;
        }
    }
    return change;
}

Result ApplyFlowgraphSnapshot(Flowgraph& flowgraph,
                              const FlowgraphSnapshot& target,
                              FlowgraphRestoreReport& report) {
    Result status = Result::SUCCESS;
    const auto step = [&status](Result result) {
        if (Failed(result) && status == Result::SUCCESS) {
            status = result;
        }
    };

    FlowgraphSnapshot current;
    JST_CHECK(Capture(flowgraph, current));

    std::unordered_map<std::string, std::vector<std::string>> renameSources;
    std::unordered_map<std::string, std::vector<std::string>> renameTargets;
    for (const auto& added : target.blocks) {
        if (current.find(added.name)) {
            continue;
        }
        for (const auto& removed : current.blocks) {
            if (!target.find(removed.name) && added.sameRuntime(removed) && added.configHash == removed.configHash) {
                renameSources[added.name].push_back(removed.name);
                renameTargets[removed.name].push_back(added.name);
            }
        }
    }
    for (const auto& [added, sources] : renameSources) {
        if (sources.size() != 1 || renameTargets.at(sources.front()).size() != 1) {
            continue;
        }
        const Result result = flowgraph.blockRename(sources.front(), added);
        step(result);
        if (Applied(result)) {
            report.renamed.emplace_back(sources.front(), added);
        }
    }
    if (!report.renamed.empty()) {
        JST_CHECK(Capture(flowgraph, current));
    }

    // Unlink first, as an edge that's going away could make a new one look like a cycle.
    for (const auto& block : current.blocks) {
        const auto* wanted = target.find(block.name);
        if (!wanted) {
            continue;
        }
        for (const auto& [slot, _] : block.inputs) {
            if (!block.sameInput(slot, *wanted)) {
                step(flowgraph.blockDisconnect(block.name, slot));
            }
        }
    }

    bool destroyed = false;
    const auto currentOrder = DependencyOrder(current);
    for (auto block = currentOrder.rbegin(); block != currentOrder.rend(); ++block) {
        const auto* wanted = target.find((*block)->name);
        if (wanted && wanted->type == (*block)->type) {
            continue;
        }
        const Result result = flowgraph.blockDestroy((*block)->name);
        step(result);
        if (Applied(result)) {
            report.removed.push_back((*block)->name);
            destroyed = true;
        }
    }
    if (destroyed) {
        JST_CHECK(Capture(flowgraph, current));
    }

    for (const auto* block : DependencyOrder(target)) {
        const auto* existing = current.find(block->name);
        if (!existing) {
            std::vector<std::string> stale;
            JST_CHECK(flowgraph.metadata().keys(stale, block->name));
            for (const auto& key : stale) {
                step(flowgraph.metadata().clear(key, block->name));
            }
            for (const auto& [key, value] : block->metadata) {
                step(flowgraph.metadata().set(key, std::any_cast<const Parser::Map&>(value), block->name));
            }
            if (block->node.has_value()) {
                step(flowgraph.metadata().set("node", *block->node, block->name));
            }

            TensorMap inputs;
            for (const auto& [slot, source] : block->inputs) {
                inputs[slot].requested(source.block, source.port);
            }
            step(flowgraph.blockCreate(block->name,
                                       block->type,
                                       block->config,
                                       inputs,
                                       block->device,
                                       block->runtime,
                                       block->provider,
                                       block->frozen));
            continue;
        }

        if (!block->sameRuntime(*existing)) {
            step(flowgraph.blockRecreate(block->name,
                                         block->config,
                                         block->device,
                                         block->runtime,
                                         block->provider));
        } else if (block->configHash != existing->configHash) {
            step(flowgraph.blockReconfigure(block->name, block->config));
        }

        if (block->frozen != existing->frozen) {
            step(flowgraph.blockFreeze(block->name, block->frozen));
        }

        for (const auto& [slot, source] : block->inputs) {
            if (!block->sameInput(slot, *existing)) {
                step(flowgraph.blockConnect(block->name, slot, source.block, source.port));
            }
        }

        if (block->node.has_value()) {
            step(flowgraph.metadata().set("node", *block->node, block->name));
        }
    }

    return status;
}

}  // namespace Jetstream
