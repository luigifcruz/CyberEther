#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_STACKS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_STACKS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

#include <any>
#include <cmath>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct StackActions {
    using Filter = std::tuple<MailCreateStack,
                              MailDeleteStack,
                              MailSetStackGeometry,
                              MailSetStackLayout,
                              MailSetSurfaceDetached,
                              MailSetSurfaceConfigCollapsed,
                              MailSetConfigDetached,
                              MailRemoveStackBlock,
                              MailRenameStackBlock>;
    using FlowgraphState = DefaultCompositorState::FlowgraphState;
    using StackWindowState = FlowgraphState::StackWindowState;

    FlowgraphState& state;
    DefaultCompositorCallbacks& callbacks;

    StackActions(FlowgraphState& state,
                 DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    void restoreFromMetadata() {
        for (const auto& [flowgraphId, flowgraph] : state.items) {
            if (!flowgraph) {
                continue;
            }

            if (state.stacks.contains(flowgraphId)) {
                continue;
            }

            if (!flowgraph->metadata().has("stacks")) {
                continue;
            }

            Parser::Map stackMap;
            if (flowgraph->metadata().get("stacks", stackMap) != Result::SUCCESS) {
                JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to load stack metadata for flowgraph '{}'.", flowgraphId);
                continue;
            }

            std::unordered_map<std::string, StackWindowState> stacks;
            for (const auto& [stackId, encodedStack] : stackMap) {
                if (stackId.empty() || encodedStack.type() != typeid(Parser::Map)) {
                    continue;
                }

                StackMeta meta;
                if (meta.deserialize(std::any_cast<const Parser::Map&>(encodedStack)) != Result::SUCCESS) {
                    JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to decode stack '{}' for flowgraph '{}'.", stackId, flowgraphId);
                    continue;
                }
                if (meta.title.empty()) {
                    meta.title = stackId;
                }
                const bool restoreDockLayout = meta.layout.has_value();

                stacks[stackId] = StackWindowState{
                    .meta = std::move(meta),
                    .restoreDockLayout = restoreDockLayout,
                    .dockInMainDockspace = true,
                };
            }

            state.stacks.emplace(flowgraphId, std::move(stacks));
        }
    }

    Result persistFlowgraphStacks(const std::string& flowgraphId) {
        if (!state.items.contains(flowgraphId)) {
            JST_ERROR("Failed to persist stacks because flowgraph was not found.");
            return Result::ERROR;
        }

        Parser::Map serializedStacks;
        auto stacksIt = state.stacks.find(flowgraphId);
        if (stacksIt != state.stacks.end()) {
            for (auto& [stackId, stack] : stacksIt->second) {
                if (stackId.empty()) {
                    continue;
                }
                if (stack.meta.title.empty()) {
                    stack.meta.title = stackId;
                }

                Parser::Map stackData;
                JST_CHECK(stack.meta.serialize(stackData));
                serializedStacks[stackId] = std::move(stackData);
            }
        }

        return state.items.at(flowgraphId)->metadata().set("stacks", serializedStacks);
    }

    Result handle(const MailCreateStack& msg) {
        if (!state.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to create stack because flowgraph was not found.");
            return Result::ERROR;
        }

        auto& stacks = state.stacks[msg.flowgraph];
        U64 index = 0;
        std::string stackId;
        do {
            stackId = jst::fmt::format("stack_{}", index++);
        } while (stacks.contains(stackId));

        const U64 stackNumber = index - 1;
        StackMeta meta;
        meta.title = jst::fmt::format("Stack {}", stackNumber);
        meta.x = 80.0f + static_cast<F32>(stackNumber) * 24.0f;
        meta.y = 80.0f + static_cast<F32>(stackNumber) * 24.0f;
        meta.width = 500.0f;
        meta.height = 300.0f;

        stacks[stackId] = StackWindowState{
            .meta = std::move(meta),
            .restoreDockLayout = false,
            .dockInMainDockspace = true,
        };

        JST_CHECK(persistFlowgraphStacks(msg.flowgraph));
        callbacks.notify(Sakura::ToastType::Success, 3000, "New stack created.");

        return Result::SUCCESS;
    }

    Result handle(const MailDeleteStack& msg) {
        if (!state.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto stacksIt = state.stacks.find(msg.flowgraph);
        if (stacksIt != state.stacks.end()) {
            stacksIt->second.erase(msg.stackId);
        }

        return persistFlowgraphStacks(msg.flowgraph);
    }

    Result handle(const MailSetStackGeometry& msg) {
        auto* stack = findStack(msg.flowgraph, msg.stackId);
        if (!stack) {
            return Result::SUCCESS;
        }

        if (sameGeometry(stack->meta, msg)) {
            return Result::SUCCESS;
        }

        stack->meta.x = msg.x;
        stack->meta.y = msg.y;
        stack->meta.width = msg.width;
        stack->meta.height = msg.height;
        return persistFlowgraphStacks(msg.flowgraph);
    }

    Result handle(const MailSetStackLayout& msg) {
        auto* stack = findStack(msg.flowgraph, msg.stackId);
        if (!stack) {
            return Result::SUCCESS;
        }

        const bool changed = Parser::Hash(stack->meta.layout) != Parser::Hash(msg.layout);
        stack->restoreDockLayout = false;
        if (!changed) {
            return Result::SUCCESS;
        }

        stack->meta.layout = msg.layout;
        return persistFlowgraphStacks(msg.flowgraph);
    }

    Result handle(const MailSetSurfaceDetached& msg) {
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
        if (meta.detached == msg.detached) {
            return Result::SUCCESS;
        }

        meta.detached = msg.detached;
        if (meta.windowId.empty()) {
            meta.windowId = makeSurfaceWindowId(*flowgraph, msg.block, msg.surface);
        }
        return flowgraph->metadata().set(metaKey, meta, msg.block);
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

        if (!msg.detached) {
            JST_CHECK(editStackLayouts(msg.flowgraph, [&msg](StackDockLayoutMeta& layout) {
                return removeBlockFromLayout(layout, msg.block, false);
            }));
        }

        ConfigMeta meta;
        JST_CHECK(flowgraph->metadata().get("config", meta, msg.block));
        if (meta.detached == msg.detached) {
            return Result::SUCCESS;
        }

        meta.detached = msg.detached;
        if (meta.windowId.empty()) {
            meta.windowId = makeConfigWindowId(*flowgraph, msg.block);
        }
        return flowgraph->metadata().set("config", meta, msg.block);
    }

    Result handle(const MailRemoveStackBlock& msg) {
        return editStackLayouts(msg.flowgraph, [&msg](StackDockLayoutMeta& layout) {
            return removeBlockFromLayout(layout, msg.block, true);
        });
    }

    Result handle(const MailRenameStackBlock& msg) {
        if (msg.oldId == msg.newId) {
            return Result::SUCCESS;
        }

        return editStackLayouts(msg.flowgraph, [&msg](StackDockLayoutMeta& layout) {
            return renameBlockInLayout(layout, msg.oldId, msg.newId);
        }, true);
    }

 private:
    static std::string makeConfigWindowId(Flowgraph& flowgraph, const std::string& block) {
        std::unordered_set<std::string> usedIds;
        forEachBlock(flowgraph, [&](const std::string& other) {
            ConfigMeta meta;
            if (flowgraph.metadata().get("config", meta, other) == Result::SUCCESS) {
                usedIds.insert(meta.windowId);
            }
        });
        return makeWindowId(block, usedIds);
    }

    static std::string makeSurfaceWindowId(Flowgraph& flowgraph,
                                           const std::string& block,
                                           const std::string& surface) {
        std::unordered_set<std::string> usedIds;
        forEachBlock(flowgraph, [&](const std::string& other) {
            std::vector<std::string> keys;
            flowgraph.metadata().keys(keys, other);
            for (const auto& key : keys) {
                SurfaceMeta meta;
                if (key.starts_with("surface_") &&
                    flowgraph.metadata().get(key, meta, other) == Result::SUCCESS) {
                    usedIds.insert(meta.windowId);
                }
            }
        });
        return makeWindowId(block + ":" + surface, usedIds);
    }

    static void forEachBlock(Flowgraph& flowgraph, const std::function<void(const std::string&)>& visit) {
        std::vector<std::string> blocks;
        flowgraph.view().keys(blocks);
        for (const auto& block : blocks) {
            visit(block);
        }
    }

    // Keep the window identity in block metadata, which follows a rename.
    // Reserve existing identities so reusing an old block name is safe.
    static std::string makeWindowId(const std::string& base, const std::unordered_set<std::string>& usedIds) {
        U64 suffix = 0;
        std::string windowId;
        do {
            windowId = base + ":" + std::to_string(suffix++);
        } while (usedIds.contains(windowId));
        return windowId;
    }

    Result editStackLayouts(const std::string& flowgraphId,
                            const std::function<bool(StackDockLayoutMeta&)>& edit,
                            const bool restoreDockLayout = false) {
        if (!state.items.contains(flowgraphId)) {
            return Result::SUCCESS;
        }

        auto stacksIt = state.stacks.find(flowgraphId);
        if (stacksIt == state.stacks.end()) {
            return Result::SUCCESS;
        }

        bool changed = false;
        for (auto& [_, stack] : stacksIt->second) {
            if (!stack.meta.layout.has_value() || !edit(*stack.meta.layout)) {
                continue;
            }
            compactLayout(*stack.meta.layout);
            if (emptyLayout(*stack.meta.layout)) {
                stack.meta.layout.reset();
            }
            stack.restoreDockLayout |= restoreDockLayout;
            changed = true;
        }

        return changed ? persistFlowgraphStacks(flowgraphId) : Result::SUCCESS;
    }

    static bool renameBlockInLayout(StackDockLayoutMeta& layout,
                                    const std::string& oldId,
                                    const std::string& newId) {
        bool changed = false;
        if (layout.configs.has_value()) {
            for (auto& config : *layout.configs) {
                if (config.block == oldId) {
                    config.block = newId;
                    changed = true;
                }
            }
        }
        if (layout.surfaces.has_value()) {
            for (auto& surface : *layout.surfaces) {
                if (surface.block == oldId) {
                    surface.block = newId;
                    changed = true;
                }
            }
        }
        if (layout.children.has_value()) {
            for (auto& child : *layout.children) {
                changed |= renameBlockInLayout(child, oldId, newId);
            }
        }
        return changed;
    }

    static bool removeBlockFromLayout(StackDockLayoutMeta& layout,
                                      const std::string& block,
                                      const bool includeSurfaces) {
        bool changed = false;
        if (layout.configs.has_value()) {
            changed |= std::erase_if(*layout.configs, [&block](const auto& config) {
                return config.block == block;
            }) > 0;
            if (layout.configs->empty()) {
                layout.configs.reset();
            }
        }
        if (includeSurfaces && layout.surfaces.has_value()) {
            changed |= std::erase_if(*layout.surfaces, [&block](const auto& surface) {
                return surface.block == block;
            }) > 0;
            if (layout.surfaces->empty()) {
                layout.surfaces.reset();
            }
        }
        if (layout.children.has_value()) {
            for (auto& child : *layout.children) {
                changed |= removeBlockFromLayout(child, block, includeSurfaces);
            }
        }
        return changed;
    }

    static bool emptyLayout(const StackDockLayoutMeta& layout) {
        return !layout.flowgraphs.has_value() &&
               !layout.surfaces.has_value() &&
               !layout.configs.has_value() &&
               !layout.children.has_value();
    }

    static void compactLayout(StackDockLayoutMeta& layout) {
        if (!layout.children.has_value()) {
            return;
        }

        auto& children = *layout.children;
        for (auto& child : children) {
            compactLayout(child);
        }
        std::erase_if(children, emptyLayout);

        if (children.empty()) {
            layout.children.reset();
        } else if (children.size() == 1) {
            auto child = std::move(children.front());
            layout = std::move(child);
        }
    }

    StackWindowState* findStack(const std::string& flowgraphId, const std::string& stackId) {
        if (!state.items.contains(flowgraphId)) {
            return nullptr;
        }

        auto flowgraphIt = state.stacks.find(flowgraphId);
        if (flowgraphIt == state.stacks.end()) {
            return nullptr;
        }

        auto stackIt = flowgraphIt->second.find(stackId);
        if (stackIt == flowgraphIt->second.end()) {
            return nullptr;
        }

        return &stackIt->second;
    }

    static bool closeEnough(const F32 lhs, const F32 rhs) {
        return std::abs(lhs - rhs) <= 0.5f;
    }

    static bool sameGeometry(const StackMeta& meta, const MailSetStackGeometry& msg) {
        return closeEnough(meta.x, msg.x) &&
               closeEnough(meta.y, msg.y) &&
               closeEnough(meta.width, msg.width) &&
               closeEnough(meta.height, msg.height);
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_STACKS_HH
