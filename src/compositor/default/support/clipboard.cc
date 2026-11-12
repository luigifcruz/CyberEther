#include "clipboard.hh"

#include "history.hh"

#include "../model/meta.hh"

#include "jetstream/flowgraph_view.hh"
#include "jetstream/parser.hh"
#include "tools/naming.hh"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace Jetstream {

namespace {

constexpr Extent2D<F32> kDuplicateOffset = {40.0f, 40.0f};

Extent2D<F32> Shift(const FlowgraphSnapshot& blocks, const std::optional<Extent2D<F32>>& point) {
    if (!point.has_value()) {
        return kDuplicateOffset;
    }

    std::optional<Extent2D<F32>> corner;
    for (const auto& block : blocks.blocks) {
        if (!block.node.has_value()) {
            continue;
        }
        if (!corner.has_value()) {
            corner = Extent2D<F32>{block.node->x, block.node->y};
            continue;
        }
        corner->x = std::min(corner->x, block.node->x);
        corner->y = std::min(corner->y, block.node->y);
    }
    if (!corner.has_value()) {
        return {0.0f, 0.0f};
    }
    return {point->x - corner->x, point->y - corner->y};
}

template<typename T>
Result Attach(Parser::Map& metadata, const std::string& key) {
    T meta;
    if (Parser::Deserialize(metadata, key, meta) != Result::SUCCESS || !meta.detached) {
        return Result::SUCCESS;
    }
    meta.detached = false;
    return Parser::Serialize(metadata, key, meta);
}

Result Attach(Parser::Map& metadata) {
    std::vector<std::string> keys;
    for (const auto& [key, _] : metadata) {
        keys.push_back(key);
    }
    for (const auto& key : keys) {
        if (key == "config") {
            JST_CHECK(Attach<ConfigMeta>(metadata, key));
        } else if (key.starts_with("surface_")) {
            JST_CHECK(Attach<SurfaceMeta>(metadata, key));
        }
    }
    return Result::SUCCESS;
}

}  // namespace

FlowgraphSnapshot CopyFlowgraphBlocks(const FlowgraphSnapshot& source, const std::vector<std::string>& names) {
    FlowgraphSnapshot copied;
    for (const auto& block : source.blocks) {
        if (std::find(names.begin(), names.end(), block.name) != names.end()) {
            copied.blocks.push_back(block);
        }
    }
    for (auto& block : copied.blocks) {
        std::erase_if(block.inputs, [&](const auto& input) {
            return !copied.find(input.second.block);
        });
    }
    return copied;
}

Result PasteFlowgraphBlocks(Flowgraph& flowgraph,
                            const FlowgraphSnapshot& blocks,
                            std::optional<Extent2D<F32>> point,
                            std::vector<std::string>& created) {
    const Extent2D<F32> shift = Shift(blocks, point);

    std::unordered_map<std::string, std::string> names;
    std::unordered_set<std::string> assigned;
    for (const auto& block : blocks.blocks) {
        const std::string name = Naming::NextName(block.name, [&](const std::string& candidate) {
            return assigned.contains(candidate) || flowgraph.view().has(candidate);
        });
        assigned.insert(name);
        names[block.name] = name;
    }

    FlowgraphSnapshot pasted = blocks;
    for (auto& block : pasted.blocks) {
        block.name = names.at(block.name);
        for (auto& [_, source] : block.inputs) {
            source.block = names.at(source.block);
        }
        JST_CHECK(Attach(block.metadata));
        if (block.node.has_value()) {
            block.node->x += shift.x;
            block.node->y += shift.y;
        } else if (point.has_value()) {
            block.node = NodeMeta{point->x, point->y, 0.0f, 0.0f};
        }
    }

    for (const auto* block : FlowgraphDependencyOrder(pasted)) {
        JST_CHECK(CreateFlowgraphBlock(flowgraph, *block));
        created.push_back(block->name);
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
