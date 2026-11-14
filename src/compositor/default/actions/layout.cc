#include "layout.hh"

namespace Jetstream {

LayoutActions::LayoutActions(FlowgraphState& state) : state(state) {}

void LayoutActions::ReplacePane(FlowgraphState& state, const std::string& flowgraphId, const std::string& pane,
                                const TileMeta& replacement) {
    if (const auto layout = state.layouts.find(flowgraphId); layout != state.layouts.end()) {
        layout->second = ReplaceTilePane(layout->second, pane, replacement);
    }
}

void LayoutActions::KeepPanes(FlowgraphState& state, const std::string& flowgraphId,
                              const std::function<bool(const std::string&)>& keep) {
    const auto layout = state.layouts.find(flowgraphId);
    if (layout == state.layouts.end()) {
        return;
    }
    if (auto kept = FilterTiles(layout->second, keep)) {
        layout->second = std::move(*kept);
    }
}

std::optional<TileMeta> LayoutActions::TakePanes(FlowgraphState& state, const std::string& flowgraphId,
                                                 const std::function<bool(const std::string&)>& take) {
    const auto layout = state.layouts.find(flowgraphId);
    if (layout == state.layouts.end()) {
        return std::nullopt;
    }
    auto taken = FilterTiles(layout->second, take);
    KeepPanes(state, flowgraphId, [&take](const std::string& pane) {
        return !take(pane);
    });
    return taken;
}

void LayoutActions::restoreFromMetadata() {
    for (const auto& [flowgraphId, flowgraph] : state.items) {
        if (!flowgraph || state.restoredLayouts.contains(flowgraphId)) {
            continue;
        }

        auto& metadata = flowgraph->metadata();
        const bool hasLayout = metadata.has("layout");
        const bool hasDescription = !Text::IsBlank(flowgraph->description());
        if (!hasLayout && !hasDescription) {
            continue;
        }

        state.restoredLayouts.insert(flowgraphId);
        if (hasLayout) {
            TileMeta layout;
            if (metadata.get("layout", layout) == Result::SUCCESS && IsTileLayout(layout)) {
                if (auto known = FilterTiles(layout, [&](const std::string& pane) {
                    return IsLayoutPane(pane) && (pane != kLayoutDescriptionPane || hasDescription);
                })) {
                    state.layouts[flowgraphId] = std::move(*known);
                } else {
                    state.layouts.erase(flowgraphId);
                }
            } else {
                JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to load layout metadata for flowgraph '{}'.", flowgraphId);
            }
        }

        if (const auto layout = state.layouts.find(flowgraphId); layout != state.layouts.end() &&
            ContainsTilePane(layout->second, kLayoutEnvironmentPane)) {
            state.environment.try_emplace(flowgraphId);
        }
        if (persistLayout(flowgraphId) != Result::SUCCESS) {
            JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to save layout metadata for flowgraph '{}'.", flowgraphId);
        }
    }
}

Result LayoutActions::prepareSave(const std::string& flowgraphId) {
    // Old files may have detached panes but no layout metadata. Their
    // initial layout stays provisional while loading, until an explicit save.
    restoreFromMetadata();
    state.restoredLayouts.insert(flowgraphId);
    return persistLayout(flowgraphId);
}

Result LayoutActions::handle(const MailSetFlowgraphLayout& msg) {
    if (!state.items.contains(msg.flowgraph) || !msg.tiles.has_value()) {
        return Result::SUCCESS;
    }
    auto tiles = state.documentation.contains(msg.flowgraph) ? msg.tiles
                                                             : WithoutPane(*msg.tiles, kLayoutDocumentationPane);
    if (tiles.has_value() && !state.environment.contains(msg.flowgraph)) {
        tiles = WithoutPane(*tiles, kLayoutEnvironmentPane);
    }
    if (tiles.has_value() && (state.closedDescriptions.contains(msg.flowgraph) ||
        Text::IsBlank(state.items.at(msg.flowgraph)->description()))) {
        tiles = WithoutPane(*tiles, kLayoutDescriptionPane);
    }
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

Result LayoutActions::handle(const MailSetSurfaceDetached& msg) {
    if (!state.items.contains(msg.flowgraph)) {
        return Result::SUCCESS;
    }

    auto flowgraph = state.items.at(msg.flowgraph);
    if (!flowgraph->view().has(msg.block)) {
        return Result::SUCCESS;
    }

    const std::string key = MakeSurfaceTileKey(msg.block, msg.surface);
    if (msg.detached) {
        JST_CHECK(trackLayout(msg.flowgraph));
    } else {
        JST_CHECK(editLayout(msg.flowgraph, [&key](const TileMeta& tiles) {
            return FilterTiles(tiles, [&](const std::string& pane) {
                return pane != key;
            });
        }));
    }
    return setSurfaceDetachedMeta(*flowgraph, msg.block, msg.surface, msg.detached);
}

Result LayoutActions::handle(const MailSetSurfaceConfigCollapsed& msg) {
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

Result LayoutActions::handle(const MailSetConfigDetached& msg) {
    if (!state.items.contains(msg.flowgraph)) {
        return Result::SUCCESS;
    }

    auto flowgraph = state.items.at(msg.flowgraph);
    if (!flowgraph->view().has(msg.block)) {
        return Result::SUCCESS;
    }

    const std::string key = MakeConfigTileKey(msg.block);
    if (msg.detached) {
        JST_CHECK(trackLayout(msg.flowgraph));
    } else {
        JST_CHECK(editLayout(msg.flowgraph, [&key](const TileMeta& tiles) {
            return FilterTiles(tiles, [&](const std::string& pane) {
                return pane != key;
            });
        }));
    }
    return setConfigDetachedMeta(*flowgraph, msg.block, msg.detached);
}

Result LayoutActions::handle(const MailRemoveLayoutBlock& msg) {
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

Result LayoutActions::handle(const MailRenameLayoutBlock& msg) {
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

Result LayoutActions::handle(const MailOpenDocumentation& msg) {
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
    return trackLayout(msg.flowgraph);
}

Result LayoutActions::handle(const MailCloseDocumentation& msg) {
    state.documentation.erase(msg.flowgraph);
    return editLayout(msg.flowgraph, [](const TileMeta& tiles) {
        return WithoutPane(tiles, kLayoutDocumentationPane);
    });
}

Result LayoutActions::handle(const MailSetFlowgraphEnvironmentVisible& msg) {
    if (!state.items.contains(msg.flowgraph)) {
        return Result::SUCCESS;
    }
    if (msg.value) {
        state.environment.try_emplace(msg.flowgraph);
        return trackLayout(msg.flowgraph);
    }
    state.environment.erase(msg.flowgraph);
    return editLayout(msg.flowgraph, [](const TileMeta& tiles) {
        return WithoutPane(tiles, kLayoutEnvironmentPane);
    });
}

Result LayoutActions::handle(const MailCloseFlowgraphDescription& msg) {
    if (!state.items.contains(msg.flowgraph)) {
        return Result::SUCCESS;
    }
    state.closedDescriptions.insert(msg.flowgraph);
    return editLayout(msg.flowgraph, [](const TileMeta& tiles) {
        return WithoutPane(tiles, kLayoutDescriptionPane);
    });
}

Result LayoutActions::handle(const MailSetFlowgraphEnvironmentSearch& msg) {
    const auto entry = state.environment.find(msg.flowgraph);
    if (entry != state.environment.end()) {
        entry->second = msg.value;
    }
    return Result::SUCCESS;
}

std::optional<TileMeta> LayoutActions::WithoutPane(const TileMeta& tiles, const std::string& pane) {
    if (!ContainsTilePane(tiles, pane)) {
        return tiles;
    }
    return FilterTiles(tiles, [&pane](const std::string& candidate) {
        return candidate != pane;
    });
}

std::optional<TileMeta> LayoutActions::Persistent(const TileMeta& tiles) {
    std::vector<std::string> panes;
    CollectTilePanes(tiles, panes);
    if (std::none_of(panes.begin(), panes.end(), IsTransientPane)) {
        return tiles;
    }
    return FilterTiles(tiles, [](const std::string& pane) {
        return !IsTransientPane(pane);
    });
}

Result LayoutActions::persistLayout(const std::string& flowgraphId) {
    if (!state.items.contains(flowgraphId) || !state.restoredLayouts.contains(flowgraphId)) {
        return Result::SUCCESS;
    }
    auto& metadata = state.items.at(flowgraphId)->metadata();
    const auto layout = state.layouts.find(flowgraphId);
    const auto saved = layout == state.layouts.end() ? std::nullopt : Persistent(layout->second);
    if (!saved.has_value()) {
        return metadata.has("layout") ? metadata.clear("layout") : Result::SUCCESS;
    }
    return metadata.set("layout", *saved);
}

Result LayoutActions::trackLayout(const std::string& flowgraphId) {
    state.restoredLayouts.insert(flowgraphId);
    return Result::SUCCESS;
}

Result LayoutActions::setSurfaceDetachedMeta(Flowgraph& flowgraph, const std::string& block, const std::string& surface,
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

Result LayoutActions::setConfigDetachedMeta(Flowgraph& flowgraph, const std::string& block, const bool detached) {
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

Result LayoutActions::editLayout(const std::string& flowgraphId,
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

}  // namespace Jetstream
