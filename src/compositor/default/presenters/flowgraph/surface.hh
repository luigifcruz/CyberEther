#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH

#include "labels.hh"
#include "config.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../model/meta.hh"
#include "../../views/flowgraph/surface.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/parser.hh"
#include "jetstream/runtime_context.hh"

#include <any>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedSurfacePresenter {
    const PresenterContext& context;

    explicit FlowgraphDetachedSurfacePresenter(const PresenterContext& context) : context(context) {}

    std::vector<FlowgraphDetachedSurface::Config> build(const std::string& flowgraphId,
                                                        const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto referencedSurfaces = buildReferencedSurfaces(flowgraphId);
        std::vector<FlowgraphDetachedSurface::Config> configs;

        if (!flowgraph) {
            return configs;
        }

        const auto blocks = context.state.flowgraph.blocks.find(flowgraphId);
        if (blocks == context.state.flowgraph.blocks.end()) {
            return configs;
        }

        for (const auto& [blockName, blockData] : blocks->second) {
            for (const auto& surface : blockData.surfaces) {
                if (!surface) {
                    continue;
                }
                for (const auto& manifest : surface->manifests()) {
                    if (!manifest.surface) {
                        continue;
                    }

                    const std::string surfaceMetaKey = "surface_" + manifest.id;
                    SurfaceMeta surfaceMeta;
                    flowgraph->metadata().get(surfaceMetaKey, surfaceMeta, blockName);

                    if (!surfaceMeta.detached && !referencedSurfaces.contains({blockName, manifest.id})) {
                        continue;
                    }

                    const std::string windowId = MakeDetachedSurfaceWindowId(flowgraphId,
                                                                             blockName,
                                                                             manifest.id,
                                                                             surfaceMeta);

                    configs.push_back({
                        .id = windowId,
                        .title = MakeDetachedSurfaceWindowTitle(blockName, blockData.title),
                        .logicalSize = {
                            static_cast<F32>(surfaceMeta.detachedWidth),
                            static_cast<F32>(surfaceMeta.detachedHeight),
                        },
                        .configFields = BuildFlowgraphDetachedConfigFields(enqueue,
                                                                           context.state.flowgraph,
                                                                           flowgraphId,
                                                                           blockName,
                                                                           windowId,
                                                                           blockData),
                        .configOpen = !surfaceMeta.configCollapsed,
                        .texture = manifest.surface,
                        .onSize = [enqueue,
                                   surface,
                                   flowgraphId,
                                   surfaceMetaKey,
                                   blockName](const Sakura::SurfaceResize& resize) {
                            enqueue(MailResizeSurface{
                                .surface = surface,
                                .flowgraph = flowgraphId,
                                .block = blockName,
                                .metaKey = surfaceMetaKey,
                                .placement = SurfacePlacement::Detached,
                                .resize = {
                                    .logicalSize = resize.logicalSize,
                                    .framebufferSize = resize.framebufferSize,
                                    .scale = resize.scale,
                                },
                            });
                        },
                        .onInput = [enqueue, surface](InputEvent event) {
                            enqueue(MailSurfaceInput{
                                .surface = surface,
                                .event = event,
                            });
                        },
                        .onResolveCursor = [surface]() {
                            return surface->cursor();
                        },
                        .onClose = [enqueue, flowgraphId, blockName, surfaceId = manifest.id]() {
                            enqueue(MailSetSurfaceDetached{
                                .flowgraph = flowgraphId,
                                .block = blockName,
                                .surface = surfaceId,
                                .detached = false,
                            });
                        },
                        .onToggleConfigOpen = [enqueue, flowgraphId, blockName, surfaceId = manifest.id](const bool open) {
                            enqueue(MailSetSurfaceConfigCollapsed{
                                .flowgraph = flowgraphId,
                                .block = blockName,
                                .surface = surfaceId,
                                .collapsed = !open,
                            });
                        },
                    });
                }
            }
        }

        return configs;
    }

 private:
    std::set<std::pair<std::string, std::string>> buildReferencedSurfaces(const std::string& flowgraphId) const {
        std::set<std::pair<std::string, std::string>> referenced;
        const auto stacksIt = context.state.flowgraph.stacks.find(flowgraphId);
        if (stacksIt == context.state.flowgraph.stacks.end()) {
            return referenced;
        }

        for (const auto& [_, stack] : stacksIt->second) {
            if (!stack.meta.layout.has_value()) {
                continue;
            }
            collectReferencedSurfaces(*stack.meta.layout, referenced);
        }
        return referenced;
    }

    static void collectReferencedSurfaces(const StackDockLayoutMeta& layout,
                                          std::set<std::pair<std::string, std::string>>& referenced) {
        if (layout.surfaces.has_value()) {
            for (const auto& surface : *layout.surfaces) {
                if (surface.block.empty() || surface.surface.empty()) {
                    continue;
                }
                referenced.emplace(surface.block, surface.surface);
            }
        }
        if (!layout.children.has_value()) {
            return;
        }
        for (const auto& child : *layout.children) {
            collectReferencedSurfaces(child, referenced);
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH
