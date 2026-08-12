#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_STACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_STACK_HH

#include "labels.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/flowgraph/stack.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/render/sakura/base.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

inline std::string MakeStackSurfaceItemKey(const std::string& block, const std::string& surface) {
    return "surface:" + block + ":" + surface;
}

inline std::optional<std::pair<std::string, std::string>> ParseStackSurfaceItemKey(const std::string& key) {
    static const std::string prefix = "surface:";
    if (key.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }

    const std::string rest = key.substr(prefix.size());
    const std::size_t delimiter = rest.find(':');
    if (delimiter == std::string::npos || delimiter == 0 || delimiter + 1 >= rest.size()) {
        return std::nullopt;
    }

    return std::make_pair(rest.substr(0, delimiter), rest.substr(delimiter + 1));
}

inline std::optional<Sakura::DockspaceWindow::DockLayout::Direction> ToSakuraDockDirection(const std::string& direction) {
    if (direction == "left") {
        return Sakura::DockspaceWindow::DockLayout::Direction::Left;
    }
    if (direction == "right") {
        return Sakura::DockspaceWindow::DockLayout::Direction::Right;
    }
    if (direction == "up") {
        return Sakura::DockspaceWindow::DockLayout::Direction::Up;
    }
    if (direction == "down") {
        return Sakura::DockspaceWindow::DockLayout::Direction::Down;
    }
    return std::nullopt;
}

inline std::optional<std::string> ToStackDockDirection(const Sakura::DockspaceWindow::DockLayout::Direction direction) {
    switch (direction) {
        case Sakura::DockspaceWindow::DockLayout::Direction::Left: return "left";
        case Sakura::DockspaceWindow::DockLayout::Direction::Right: return "right";
        case Sakura::DockspaceWindow::DockLayout::Direction::Up: return "up";
        case Sakura::DockspaceWindow::DockLayout::Direction::Down: return "down";
    }
    return std::nullopt;
}

inline std::optional<Sakura::DockspaceWindow::DockLayout> ToSakuraDockLayout(const StackDockLayoutMeta& meta) {
    Sakura::DockspaceWindow::DockLayout layout;
    if (meta.direction.has_value()) {
        layout.direction = ToSakuraDockDirection(*meta.direction);
    }
    layout.ratio = meta.ratio;

    std::vector<Sakura::DockspaceWindow::DockItem> items;
    if (meta.flowgraphs.has_value()) {
        for (const auto& flowgraph : *meta.flowgraphs) {
            items.push_back({.key = "flowgraph", .order = flowgraph.order});
        }
    }
    if (meta.surfaces.has_value()) {
        for (const auto& surface : *meta.surfaces) {
            if (surface.block.empty() || surface.surface.empty()) {
                continue;
            }
            items.push_back({
                .key = MakeStackSurfaceItemKey(surface.block, surface.surface),
                .order = surface.order,
            });
        }
    }
    if (!items.empty()) {
        layout.items = std::move(items);
    }

    if (meta.children.has_value()) {
        std::vector<Sakura::DockspaceWindow::DockLayout> children;
        for (const auto& child : *meta.children) {
            auto sakuraChild = ToSakuraDockLayout(child);
            if (sakuraChild.has_value()) {
                children.push_back(std::move(*sakuraChild));
            }
        }
        if (!children.empty()) {
            layout.children = std::move(children);
        }
    }

    if (!layout.items.has_value() && !layout.children.has_value()) {
        return std::nullopt;
    }
    return layout;
}

inline std::optional<StackDockLayoutMeta> ToStackDockLayoutMeta(const Sakura::DockspaceWindow::DockLayout& layout) {
    StackDockLayoutMeta meta;
    if (layout.direction.has_value()) {
        meta.direction = ToStackDockDirection(*layout.direction);
    }
    meta.ratio = layout.ratio;

    std::vector<StackDockFlowgraphMeta> flowgraphs;
    std::vector<StackDockSurfaceMeta> surfaces;
    if (layout.items.has_value()) {
        for (const auto& item : *layout.items) {
            if (item.key == "flowgraph") {
                flowgraphs.push_back({.order = item.order});
                continue;
            }

            auto parsedSurface = ParseStackSurfaceItemKey(item.key);
            if (!parsedSurface.has_value()) {
                continue;
            }
            surfaces.push_back({
                .block = parsedSurface->first,
                .surface = parsedSurface->second,
                .order = item.order,
            });
        }
    }
    if (!flowgraphs.empty()) {
        meta.flowgraphs = std::move(flowgraphs);
    }
    if (!surfaces.empty()) {
        meta.surfaces = std::move(surfaces);
    }

    if (layout.children.has_value()) {
        std::vector<StackDockLayoutMeta> children;
        for (const auto& child : *layout.children) {
            auto childMeta = ToStackDockLayoutMeta(child);
            if (childMeta.has_value()) {
                children.push_back(std::move(*childMeta));
            }
        }
        if (!children.empty()) {
            meta.children = std::move(children);
        }
    }

    if (!meta.flowgraphs.has_value() && !meta.surfaces.has_value() && !meta.children.has_value()) {
        return std::nullopt;
    }
    return meta;
}

struct StackPresenter {
    const PresenterContext& context;

    explicit StackPresenter(const PresenterContext& context) : context(context) {}

    std::vector<FlowgraphStackWindow::Config> build(const std::string& flowgraphId,
                                                    const std::shared_ptr<Flowgraph>& flowgraph) const {
        std::vector<FlowgraphStackWindow::Config> configs;
        if (!flowgraph) {
            return configs;
        }

        const auto stacksIt = context.state.flowgraph.stacks.find(flowgraphId);
        if (stacksIt == context.state.flowgraph.stacks.end()) {
            return configs;
        }

        std::vector<std::string> stackIds;
        stackIds.reserve(stacksIt->second.size());
        for (const auto& [stackId, _] : stacksIt->second) {
            stackIds.push_back(stackId);
        }
        std::sort(stackIds.begin(), stackIds.end());

        for (const auto& stackId : stackIds) {
            configs.push_back(buildStack(flowgraphId, flowgraph, stackId, stacksIt->second.at(stackId)));
        }

        return configs;
    }

 private:
    FlowgraphStackWindow::Config buildStack(const std::string& flowgraphId,
                                            const std::shared_ptr<Flowgraph>& flowgraph,
                                            const std::string& stackId,
                                            const DefaultCompositorState::FlowgraphState::StackWindowState& stack) const {
        const auto enqueue = context.callbacks.enqueueMail;
        std::optional<Sakura::DockspaceWindow::DockLayout> layout;
        if (stack.meta.layout.has_value()) {
            layout = ToSakuraDockLayout(*stack.meta.layout);
        }

        return FlowgraphStackWindow::Config{
            .id = MakeStackWindowId(flowgraphId, stackId),
            .title = MakeStackWindowTitle(stackId, stack.meta),
            .position = {stack.meta.x, stack.meta.y},
            .size = {stack.meta.width, stack.meta.height},
            .parentDockId = stack.dockInMainDockspace ? std::optional<U64>{Sakura::DockspaceId()} : std::nullopt,
            .dockIntoParent = stack.dockInMainDockspace,
            .restoreLayout = stack.restoreDockLayout,
            .layout = layout,
            .dockables = buildDockables(flowgraphId, flowgraph),
            .onGeometry = [enqueue, flowgraphId, stackId](Extent2D<F32> position, Extent2D<F32> size) {
                enqueue(MailSetStackGeometry{
                    .flowgraph = flowgraphId,
                    .stackId = stackId,
                    .x = position.x,
                    .y = position.y,
                    .width = size.x,
                    .height = size.y,
                });
            },
            .onLayout = [enqueue, flowgraphId, stackId](std::optional<Sakura::DockspaceWindow::DockLayout> layout) {
                std::optional<StackDockLayoutMeta> meta;
                if (layout.has_value()) {
                    meta = ToStackDockLayoutMeta(*layout);
                }
                enqueue(MailSetStackLayout{
                    .flowgraph = flowgraphId,
                    .stackId = stackId,
                    .layout = meta,
                });
            },
            .onClose = [enqueue, flowgraphId, stackId]() {
                enqueue(MailDeleteStack{.flowgraph = flowgraphId, .stackId = stackId});
            },
        };
    }

    std::vector<Sakura::DockspaceWindow::DockableWindow> buildDockables(const std::string& flowgraphId,
                                                                       const std::shared_ptr<Flowgraph>& flowgraph) const {
        std::vector<Sakura::DockspaceWindow::DockableWindow> dockables;
        dockables.push_back({
            .key = "flowgraph",
            .label = MakeFlowgraphWindowLabel(flowgraphId, flowgraph),
        });

        std::vector<std::string> blocks;
        if (flowgraph->view().keys(blocks) != Result::SUCCESS) {
            return dockables;
        }

        for (const auto& blockName : blocks) {
            Flowgraph::View::BlockData blockData;
            if (flowgraph->view().block(blockName, blockData) != Result::SUCCESS) {
                continue;
            }

            for (const auto& surface : blockData.surfaces) {
                if (!surface) {
                    continue;
                }
                for (const auto& manifest : surface->manifests()) {
                    if (!manifest.surface) {
                        continue;
                    }
                    dockables.push_back({
                        .key = MakeStackSurfaceItemKey(blockName, manifest.id),
                        .label = MakeDetachedSurfaceWindowLabel(flowgraphId,
                                                                blockName,
                                                                manifest.id,
                                                                blockData.title),
                    });
                }
            }
        }

        return dockables;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_STACK_HH
