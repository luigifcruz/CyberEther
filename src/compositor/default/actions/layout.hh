#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LAYOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LAYOUT_HH

#include "../model/messages.hh"
#include "../model/state.hh"
#include "../model/tiles.hh"

#include "jetstream/logger.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct LayoutActions {
    using Filter = std::tuple<MailSetFlowgraphLayout,
                              MailSetSurfaceDetached,
                              MailSetSurfaceConfigCollapsed,
                              MailSetConfigDetached,
                              MailRemoveLayoutBlock,
                              MailRenameLayoutBlock,
                              MailOpenDocumentation,
                              MailCloseDocumentation>;
    using FlowgraphState = DefaultCompositorState::FlowgraphState;

    FlowgraphState& state;

    explicit LayoutActions(FlowgraphState& state) : state(state) {}

    void restoreFromMetadata() {
        for (const auto& [flowgraphId, flowgraph] : state.items) {
            if (!flowgraph || state.restoredLayouts.contains(flowgraphId)) {
                continue;
            }

            auto& metadata = flowgraph->metadata();
            const bool hasLayout = metadata.has("layout");
            const bool hasStacks = metadata.has("stacks");
            if (!hasLayout && !hasStacks) {
                continue;
            }

            state.restoredLayouts.insert(flowgraphId);
            if (hasStacks && metadata.clear("stacks") != Result::SUCCESS) {
                JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to clear stack metadata for flowgraph '{}'.", flowgraphId);
            }

            if (hasLayout) {
                TileMeta layout;
                if (metadata.get("layout", layout) == Result::SUCCESS) {
                    if (auto known = FilterTiles(layout, IsLayoutPane)) {
                        state.layouts[flowgraphId] = std::move(*known);
                    } else {
                        state.layouts.erase(flowgraphId);
                    }
                } else {
                    JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to load layout metadata for flowgraph '{}'.", flowgraphId);
                }
            }

            placeOrphans(flowgraphId);
            if (persistLayout(flowgraphId) != Result::SUCCESS) {
                JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to save layout metadata for flowgraph '{}'.", flowgraphId);
            }
        }
    }

    Result handle(const MailSetFlowgraphLayout& msg) {
        if (!state.items.contains(msg.flowgraph) || !msg.tiles.has_value()) {
            return Result::SUCCESS;
        }
        auto tiles = state.documentation.contains(msg.flowgraph) ? msg.tiles : WithoutDocumentation(*msg.tiles);
        if (!tiles.has_value()) {
            return Result::SUCCESS;
        }
        const bool known = state.layouts.contains(msg.flowgraph);
        auto& layout = state.layouts[msg.flowgraph];
        const bool changed = !known || Parser::Hash(layout) != Parser::Hash(*tiles);
        layout = std::move(*tiles);
        if (!state.restoredLayouts.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }
        if (!changed && state.items.at(msg.flowgraph)->metadata().has("layout")) {
            return Result::SUCCESS;
        }
        return persistLayout(msg.flowgraph);
    }

    Result handle(const MailSetSurfaceDetached& msg) {
        if (!state.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto flowgraph = state.items.at(msg.flowgraph);
        if (!flowgraph->view().has(msg.block)) {
            return Result::SUCCESS;
        }

        const std::string key = MakeSurfaceTileKey(msg.block, msg.surface);
        if (msg.detached) {
            JST_CHECK(place(msg.flowgraph, key));
        } else {
            JST_CHECK(editLayout(msg.flowgraph, [&key](const TileMeta& tiles) {
                return FilterTiles(tiles, [&](const std::string& pane) {
                    return pane != key;
                });
            }));
        }
        return setSurfaceDetachedMeta(*flowgraph, msg.block, msg.surface, msg.detached);
    }

    Result handle(const MailSetSurfaceConfigCollapsed& msg) {
        if (!state.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto flowgraph = state.items.at(msg.flowgraph);
        if (!flowgraph->view().has(msg.block)) {
            return Result::SUCCESS;
        }

        const std::string metaKey = "surface_" + msg.surface;
        SurfaceMeta meta;
        JST_CHECK(flowgraph->metadata().get(metaKey, meta, msg.block));
        if (meta.configCollapsed == msg.collapsed) {
            return Result::SUCCESS;
        }

        meta.configCollapsed = msg.collapsed;
        return flowgraph->metadata().set(metaKey, meta, msg.block);
    }

    Result handle(const MailSetConfigDetached& msg) {
        if (!state.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto flowgraph = state.items.at(msg.flowgraph);
        if (!flowgraph->view().has(msg.block)) {
            return Result::SUCCESS;
        }

        const std::string key = MakeConfigTileKey(msg.block);
        if (msg.detached) {
            JST_CHECK(place(msg.flowgraph, key));
        } else {
            JST_CHECK(editLayout(msg.flowgraph, [&key](const TileMeta& tiles) {
                return FilterTiles(tiles, [&](const std::string& pane) {
                    return pane != key;
                });
            }));
        }
        return setConfigDetachedMeta(*flowgraph, msg.block, msg.detached);
    }

    Result handle(const MailRemoveLayoutBlock& msg) {
        const auto documentation = state.documentation.find(msg.flowgraph);
        const bool documented = documentation != state.documentation.end() && documentation->second.block == msg.block;
        if (documented) {
            state.documentation.erase(documentation);
        }
        return editLayout(msg.flowgraph, [&msg, documented](const TileMeta& tiles) {
            return FilterTiles(tiles, [&](const std::string& pane) {
                return TileBlock(pane) != msg.block && (!documented || pane != kLayoutDocumentationPane);
            });
        });
    }

    Result handle(const MailRenameLayoutBlock& msg) {
        if (msg.oldId == msg.newId) {
            return Result::SUCCESS;
        }

        const auto documentation = state.documentation.find(msg.flowgraph);
        if (documentation != state.documentation.end() && documentation->second.block == msg.oldId) {
            documentation->second.block = msg.newId;
        }
        return editLayout(msg.flowgraph, [&msg](const TileMeta& tiles) {
            std::unordered_set<std::string> seen;
            return MapTiles(tiles, [&](const std::string& pane) -> std::optional<std::string> {
                std::string renamed = pane;
                if (auto surface = ParseSurfaceTileKey(pane); surface && surface->first == msg.oldId) {
                    renamed = MakeSurfaceTileKey(msg.newId, surface->second);
                } else if (ParseConfigTileKey(pane) == msg.oldId) {
                    renamed = MakeConfigTileKey(msg.newId);
                }
                return seen.insert(renamed).second ? std::optional<std::string>(std::move(renamed)) : std::nullopt;
            });
        });
    }

    Result handle(const MailOpenDocumentation& msg) {
        if (!state.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }
        auto flowgraph = state.items.at(msg.flowgraph);
        Flowgraph::View::BlockData block;
        if (!flowgraph->view().has(msg.block) || flowgraph->view().block(msg.block, block) != Result::SUCCESS) {
            return Result::SUCCESS;
        }
        state.documentation[msg.flowgraph] = {
            .block = msg.block,
            .title = std::move(block.title),
            .markdown = std::move(block.description),
        };
        return place(msg.flowgraph, kLayoutDocumentationPane);
    }

    Result handle(const MailCloseDocumentation& msg) {
        state.documentation.erase(msg.flowgraph);
        return editLayout(msg.flowgraph, WithoutDocumentation);
    }

 private:
    static std::optional<TileMeta> WithoutDocumentation(const TileMeta& tiles) {
        if (!ContainsTilePane(tiles, kLayoutDocumentationPane)) {
            return tiles;
        }
        return FilterTiles(tiles, [](const std::string& pane) {
            return pane != kLayoutDocumentationPane;
        });
    }

    Result persistLayout(const std::string& flowgraphId) {
        if (!state.items.contains(flowgraphId) || !state.restoredLayouts.contains(flowgraphId)) {
            return Result::SUCCESS;
        }
        auto& metadata = state.items.at(flowgraphId)->metadata();
        const auto layout = state.layouts.find(flowgraphId);
        const auto saved = layout == state.layouts.end() ? std::nullopt : WithoutDocumentation(layout->second);
        if (!saved.has_value()) {
            return metadata.has("layout") ? metadata.clear("layout") : Result::SUCCESS;
        }
        return metadata.set("layout", *saved);
    }

    Result place(const std::string& flowgraphId, const std::string& pane) {
        state.restoredLayouts.insert(flowgraphId);
        const auto layout = state.layouts.find(flowgraphId);
        if (layout != state.layouts.end() && ContainsTilePane(layout->second, pane)) {
            return Result::SUCCESS;
        }
        const TileMeta base = layout != state.layouts.end() ? layout->second : TileMeta{.pane = kLayoutEditorPane};
        state.layouts[flowgraphId] = AppendColumn(base, pane);
        return persistLayout(flowgraphId);
    }

    void placeOrphans(const std::string& flowgraphId) {
        auto flowgraph = state.items.at(flowgraphId);
        forEachBlock(*flowgraph, [&](const std::string& block) {
            ConfigMeta config;
            if (flowgraph->metadata().get("config", config, block) == Result::SUCCESS && config.detached) {
                (void)place(flowgraphId, MakeConfigTileKey(block));
            }
            std::vector<std::string> keys;
            flowgraph->metadata().keys(keys, block);
            for (const auto& key : keys) {
                SurfaceMeta surface;
                if (key.starts_with("surface_") &&
                    flowgraph->metadata().get(key, surface, block) == Result::SUCCESS && surface.detached) {
                    (void)place(flowgraphId, MakeSurfaceTileKey(block, key.substr(std::string("surface_").size())));
                }
            }
        });
    }

    Result setSurfaceDetachedMeta(Flowgraph& flowgraph, const std::string& block, const std::string& surface,
                                  const bool detached) {
        if (!flowgraph.view().has(block)) {
            return Result::SUCCESS;
        }
        const std::string metaKey = "surface_" + surface;
        SurfaceMeta meta;
        JST_CHECK(flowgraph.metadata().get(metaKey, meta, block));
        if (meta.detached == detached) {
            return Result::SUCCESS;
        }

        meta.detached = detached;
        return flowgraph.metadata().set(metaKey, meta, block);
    }

    Result setConfigDetachedMeta(Flowgraph& flowgraph, const std::string& block, const bool detached) {
        if (!flowgraph.view().has(block)) {
            return Result::SUCCESS;
        }
        ConfigMeta meta;
        JST_CHECK(flowgraph.metadata().get("config", meta, block));
        if (meta.detached == detached) {
            return Result::SUCCESS;
        }

        meta.detached = detached;
        return flowgraph.metadata().set("config", meta, block);
    }

    static void forEachBlock(Flowgraph& flowgraph, const std::function<void(const std::string&)>& visit) {
        std::vector<std::string> blocks;
        flowgraph.view().keys(blocks);
        for (const auto& block : blocks) {
            visit(block);
        }
    }

    Result editLayout(const std::string& flowgraphId,
                      const std::function<std::optional<TileMeta>(const TileMeta&)>& edit) {
        if (!state.items.contains(flowgraphId)) {
            return Result::SUCCESS;
        }

        const auto layout = state.layouts.find(flowgraphId);
        if (layout == state.layouts.end()) {
            return Result::SUCCESS;
        }

        auto next = edit(layout->second);
        if (next.has_value() && Parser::Hash(*next) == Parser::Hash(layout->second)) {
            return Result::SUCCESS;
        }
        if (next.has_value()) {
            layout->second = std::move(*next);
        } else {
            state.layouts.erase(layout);
        }
        return persistLayout(flowgraphId);
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_LAYOUT_HH
