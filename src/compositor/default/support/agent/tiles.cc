#include "tiles.hh"

#include "../../model/callbacks.hh"
#include "../../model/meta.hh"
#include "../../model/state.hh"
#include "../../model/tiles.hh"
#include "../../presenters/flowgraph/layout.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/module_surface.hh"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace Jetstream {

namespace {

using Json = nlohmann::json;

constexpr const char* kInspect = "inspect_tiles";
constexpr const char* kArrange = "arrange_tiles";
constexpr U64 kMaxDepth = 8;
constexpr F64 kMaxSize = 100000.0;

Agent::Tools::Output Reply(const Json& body) {
    return {body.dump(-1, ' ', false, Json::error_handler_t::replace)};
}

std::string EditorTitle() {
    return "Flowgraph";
}

std::optional<TileMeta> ParseNode(const Json& node, const std::string& path, U64 depth,
                                  std::vector<std::string>& panes, std::string& error) {
    if (!node.is_object()) {
        error = path + " must be an object.";
        return std::nullopt;
    }
    if (depth > kMaxDepth) {
        error = jst::fmt::format("{} nests deeper than {} levels.", path, kMaxDepth);
        return std::nullopt;
    }

    const bool leaf = node.contains("tile");
    if (leaf == node.contains("split")) {
        error = path + " needs exactly one of 'tile' or 'split'.";
        return std::nullopt;
    }

    TileMeta meta;
    for (const auto& [key, value] : node.items()) {
        if (key == "size") {
            if (!value.is_number() || !std::isfinite(value.get<F64>()) || value.get<F64>() < 0.0 ||
                value.get<F64>() > kMaxSize) {
                error = path + ".size must be a non-negative number of logical pixels.";
                return std::nullopt;
            }
            meta.size = value.get<F32>();
        } else if (leaf && key == "tile") {
            if (!value.is_string() || value.get<std::string>().empty()) {
                error = path + ".tile must be a non-empty tile id.";
                return std::nullopt;
            }
            meta.pane = value.get<std::string>();
            panes.push_back(*meta.pane);
        } else if (leaf && key == "minimized") {
            if (!value.is_boolean()) {
                error = path + ".minimized must be a boolean.";
                return std::nullopt;
            }
            if (value.get<bool>()) {
                meta.minimized = true;
            }
        } else if (!leaf && key == "split") {
            if (!value.is_string() || (value != "row" && value != "column")) {
                error = path + ".split must be 'row' or 'column'.";
                return std::nullopt;
            }
            meta.split = value.get<std::string>();
        } else if (!leaf && key == "children") {
            if (!value.is_array() || value.empty()) {
                error = path + ".children must be a non-empty array.";
                return std::nullopt;
            }
            std::vector<TileMeta> children;
            for (U64 index = 0; index < value.size(); ++index) {
                auto child = ParseNode(value[index], jst::fmt::format("{}.children[{}]", path, index),
                                       depth + 1, panes, error);
                if (!child.has_value()) {
                    return std::nullopt;
                }
                child->size = child->size.value_or(0.0f);
                children.push_back(std::move(*child));
            }
            meta.children = std::move(children);
        } else {
            error = jst::fmt::format("{} has unknown field '{}'.", path, key);
            return std::nullopt;
        }
    }

    if (!leaf && !meta.children.has_value()) {
        error = path + " needs children.";
        return std::nullopt;
    }
    if (!leaf && meta.children->size() == 1) {
        auto single = std::move(meta.children->front());
        single.size = meta.size;
        return single;
    }
    return meta;
}

}  // namespace

struct AgentTiles::Tile {
    std::string id;
    std::string title;
    std::string block;
    std::string surface;
    bool open = false;
};

AgentTiles::AgentTiles(DefaultCompositorState& state, const DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

bool AgentTiles::Handles(const std::string& name) {
    return name == kInspect || name == kArrange;
}

bool AgentTiles::ReadOnly(const std::string& name) {
    return name == kInspect;
}

std::vector<Agent::Provider::Function> AgentTiles::Functions() {
    return {
        {
            kInspect,
            "Inspect the tile workspace of the selected graph. Tiles are panels docked beside the flowgraph "
            "editor: the editor itself ('flowgraph'), block plots ('surface:<block>:<surface>'), block "
            "controls with live metrics ('config:<block>'), and any open documentation, environment, "
            "description, or terminal panels. Returns layout, the current tile tree in the arrange_tiles "
            "format with titles, and unplaced, the block plots and controls that can still be added.",
            R"({"type":"object","properties":{},"additionalProperties":false})",
        },
        {
            kArrange,
            "Replace the tile layout of the selected graph, for example to build a dashboard or monitoring "
            "view from block plots and controls. layout is a tree. A leaf {tile, minimized} places one tile "
            "by id from inspect_tiles. A split {split, children} places its children side by side for 'row' "
            "or stacked top to bottom for 'column'. Any node can set size in logical pixels, where 0 or "
            "omitted means default. Make the root a row of columns. Root column widths default to 420, and "
            "the column holding the open flowgraph tile takes the remaining width. When that tile is "
            "minimized, the columns scale to fill the width in proportion to their sizes. Sizes inside a "
            "split are relative to their siblings. The flowgraph tile must appear, and minimized=true "
            "collapses a tile while at least one tile stays open. Plots and controls in layout open, and "
            "those left out return to their nodes. Open documentation, environment, description, and "
            "terminal tiles stay open when left out. Plots appear once their blocks exist, so create blocks "
            "first. Returns the tiles that opened and closed.",
            R"({"type":"object","required":["layout"],"properties":{"layout":{"type":"object","description":"Root node. A leaf is {\"tile\": id, \"size\": px, \"minimized\": bool}. A split is {\"split\": \"row\" or \"column\", \"children\": [nodes], \"size\": px}. size and minimized are optional."}},"additionalProperties":false})",
        },
    };
}

Agent::Tools::Output AgentTiles::execute(const std::shared_ptr<Flowgraph>& flowgraph,
                                         const std::string& name,
                                         const std::string& args) const {
    if (!flowgraph) {
        return Agent::Tools::Error("No flowgraph is selected, so there are no tiles.");
    }

    const auto& items = state.flowgraph.items;
    const auto item = std::find_if(items.begin(), items.end(), [&](const auto& entry) {
        return entry.second == flowgraph;
    });
    if (item == items.end()) {
        return Agent::Tools::Error("The selected flowgraph is not open.");
    }

    if (name == kInspect) {
        return inspect(item->first, *flowgraph);
    }
    return arrange(item->first, *flowgraph, args);
}

std::vector<AgentTiles::Tile> AgentTiles::Catalog(Flowgraph& flowgraph) {
    std::vector<Tile> tiles;
    std::vector<std::string> blocks;
    if (flowgraph.view().keys(blocks) != Result::SUCCESS) {
        return tiles;
    }

    for (const auto& block : blocks) {
        Flowgraph::View::BlockData data;
        if (flowgraph.view().block(block, data) != Result::SUCCESS) {
            continue;
        }
        for (const auto& surface : data.surfaces) {
            if (!surface) {
                continue;
            }
            for (const auto& manifest : surface->manifests()) {
                if (manifest.id.empty()) {
                    continue;
                }
                SurfaceMeta meta;
                flowgraph.metadata().get("surface_" + manifest.id, meta, block);
                tiles.push_back({
                    .id = MakeSurfaceTileKey(block, manifest.id),
                    .title = MakeSurfaceTileTitle(block, data.title),
                    .block = block,
                    .surface = manifest.id,
                    .open = meta.detached,
                });
            }
        }

        if (data.interfaceConfigs.empty()) {
            continue;
        }
        ConfigMeta meta;
        flowgraph.metadata().get("config", meta, block);
        tiles.push_back({
            .id = MakeConfigTileKey(block),
            .title = MakeConfigTileTitle(block, data.title),
            .block = block,
            .open = meta.detached,
        });
    }

    return tiles;
}

Agent::Tools::Output AgentTiles::inspect(const std::string& flowgraphId, Flowgraph& flowgraph) const {
    const auto tiles = Catalog(flowgraph);
    std::unordered_map<std::string, std::string> titles;
    for (const auto& tile : tiles) {
        titles[tile.id] = tile.title;
    }

    const auto describe = [&](const auto& self, const TileMeta& node) -> Json {
        Json out = Json::object();
        if (IsTileLeaf(node)) {
            const auto pane = node.pane.value_or(std::string{});
            const auto title = titles.find(pane);
            out["tile"] = pane;
            out["title"] = title != titles.end() ? title->second
                                                 : pane == kLayoutEditorPane ? EditorTitle() : DescribeTile(pane);
            if (node.minimized.value_or(false)) {
                out["minimized"] = true;
            }
        } else {
            out["split"] = node.split.value_or("row");
            Json children = Json::array();
            for (const auto& child : *node.children) {
                children.push_back(self(self, child));
            }
            out["children"] = std::move(children);
        }
        if (node.size.value_or(0.0f) > 0.0f) {
            out["size"] = *node.size;
        }
        return out;
    };

    std::vector<std::string> placed;
    Json layout = {{"tile", kLayoutEditorPane}, {"title", EditorTitle()}};
    if (const auto current = state.flowgraph.layouts.find(flowgraphId); current != state.flowgraph.layouts.end()) {
        layout = describe(describe, current->second);
        CollectTilePanes(current->second, placed);
    }

    Json unplaced = Json::array();
    for (const auto& tile : tiles) {
        if (std::find(placed.begin(), placed.end(), tile.id) == placed.end()) {
            unplaced.push_back({{"tile", tile.id}, {"title", tile.title}});
        }
    }

    return Reply({{"layout", std::move(layout)}, {"unplaced", std::move(unplaced)}});
}

Agent::Tools::Output AgentTiles::arrange(const std::string& flowgraphId, Flowgraph& flowgraph,
                                         const std::string& args) const {
    const auto request = Json::parse(args, nullptr, false);
    if (!request.is_object()) {
        return Agent::Tools::Error("Arguments must be a JSON object.");
    }
    for (const auto& [key, _] : request.items()) {
        if (key != "layout") {
            return Agent::Tools::Error(jst::fmt::format("Unknown field '{}'.", key));
        }
    }
    if (!request.contains("layout")) {
        return Agent::Tools::Error("'layout' is required.");
    }

    std::string error;
    std::vector<std::string> panes;
    auto tree = ParseNode(request["layout"], "layout", 0, panes, error);
    if (!tree.has_value()) {
        return Agent::Tools::Error(error);
    }
    tree->size.reset();

    const auto tiles = Catalog(flowgraph);
    std::vector<std::string> allowed = {kLayoutEditorPane};
    for (const auto& tile : tiles) {
        allowed.push_back(tile.id);
    }
    if (const auto current = state.flowgraph.layouts.find(flowgraphId); current != state.flowgraph.layouts.end()) {
        std::vector<std::string> open;
        CollectTilePanes(current->second, open);
        for (const auto& pane : open) {
            if (!TileBlock(pane).has_value() && pane != kLayoutEditorPane) {
                allowed.push_back(pane);
            }
        }
    }

    std::unordered_set<std::string> wanted;
    for (const auto& pane : panes) {
        if (!wanted.insert(pane).second) {
            return Agent::Tools::Error(jst::fmt::format("Tile '{}' appears more than once.", pane));
        }
        if (std::find(allowed.begin(), allowed.end(), pane) == allowed.end()) {
            return Agent::Tools::Error(jst::fmt::format("Unknown tile '{}'. Available tiles: {}.",
                                                        pane, jst::fmt::join(allowed, ", ")));
        }
    }
    if (!wanted.contains(kLayoutEditorPane)) {
        return Agent::Tools::Error("layout must include the 'flowgraph' tile.");
    }

    Json opened = Json::array();
    Json closed = Json::array();
    for (const auto& tile : tiles) {
        const bool open = wanted.contains(tile.id);
        if (open != tile.open) {
            (open ? opened : closed).push_back(tile.id);
        }
        if (tile.surface.empty()) {
            callbacks.enqueueMail(MailSetConfigDetached{
                .flowgraph = flowgraphId,
                .block = tile.block,
                .detached = open,
            });
        } else {
            callbacks.enqueueMail(MailSetSurfaceDetached{
                .flowgraph = flowgraphId,
                .block = tile.block,
                .surface = tile.surface,
                .detached = open,
            });
        }
    }
    callbacks.enqueueMail(MailSetFlowgraphLayout{
        .flowgraph = flowgraphId,
        .tiles = std::move(tree),
        .track = true,
    });

    return Reply({{"opened", std::move(opened)}, {"closed", std::move(closed)}});
}

}  // namespace Jetstream
