#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_STACKS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_STACKS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"
#include "jetstream/flowgraph_metadata.hh"

#include <cmath>
#include <string>
#include <tuple>
#include <utility>

namespace Jetstream {

struct StackActions {
    using Filter = std::tuple<MailCreateStack,
                              MailDeleteStack,
                              MailSetStackGeometry,
                              MailSetStackLayout,
                              MailSetSurfaceDetached>;
    using StackWindowState = DefaultCompositorState::FlowgraphState::StackWindowState;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    StackActions(DefaultCompositorState& state,
                 DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result persistFlowgraphStacks(const std::string& flowgraphId) {
        if (!state.flowgraph.items.contains(flowgraphId)) {
            JST_ERROR("Failed to persist stacks because flowgraph was not found.");
            return Result::ERROR;
        }

        Parser::Map serializedStacks;
        auto stacksIt = state.flowgraph.stacks.find(flowgraphId);
        if (stacksIt != state.flowgraph.stacks.end()) {
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

        return state.flowgraph.items.at(flowgraphId)->metadata().set("stacks", serializedStacks);
    }

    Result handle(const MailCreateStack& msg) {
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            JST_ERROR("Failed to create stack because flowgraph was not found.");
            return Result::ERROR;
        }

        auto& stacks = state.flowgraph.stacks[msg.flowgraph];
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
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto stacksIt = state.flowgraph.stacks.find(msg.flowgraph);
        if (stacksIt != state.flowgraph.stacks.end()) {
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
        if (!state.flowgraph.items.contains(msg.flowgraph)) {
            return Result::SUCCESS;
        }

        auto flowgraph = state.flowgraph.items.at(msg.flowgraph);
        const auto blocks = flowgraph->blockList();
        if (!blocks.contains(msg.block)) {
            return Result::SUCCESS;
        }

        const std::string metaKey = "surface_" + msg.surface;
        SurfaceMeta meta;
        JST_CHECK(flowgraph->metadata().get(metaKey, meta, msg.block));
        if (meta.detached == msg.detached) {
            return Result::SUCCESS;
        }

        meta.detached = msg.detached;
        return flowgraph->metadata().set(metaKey, meta, msg.block);
    }

 private:
    StackWindowState* findStack(const std::string& flowgraphId, const std::string& stackId) {
        if (!state.flowgraph.items.contains(flowgraphId)) {
            return nullptr;
        }

        auto flowgraphIt = state.flowgraph.stacks.find(flowgraphId);
        if (flowgraphIt == state.flowgraph.stacks.end()) {
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
