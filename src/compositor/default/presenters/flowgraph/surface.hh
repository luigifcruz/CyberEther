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
#include <string>
#include <unordered_set>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedSurfacePresenter {
    const PresenterContext& context;

    explicit FlowgraphDetachedSurfacePresenter(const PresenterContext& context) : context(context) {}

    std::vector<FlowgraphDetachedSurface::Config> build(const std::string& flowgraphId,
                                                        const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto referencedSurfaces = buildReferencedSurfaceIds(flowgraphId);
        std::vector<FlowgraphDetachedSurface::Config> configs;

        if (!flowgraph) {
            return configs;
        }

        std::vector<std::string> blocks;
        if (flowgraph->view().keys(blocks) != Result::SUCCESS) {
            return configs;
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

                    const std::string surfaceMetaKey = "surface_" + manifest.id;
                    SurfaceMeta surfaceMeta;
                    flowgraph->metadata().get(surfaceMetaKey, surfaceMeta, blockName);

                    const std::string windowId = MakeDetachedSurfaceWindowId(flowgraphId,
                                                                             blockName,
                                                                             manifest.id);
                    if (!surfaceMeta.detached && !referencedSurfaces.contains(windowId)) {
                        continue;
                    }

                    const auto texture = manifest.surface;
                    configs.push_back({
                        .id = windowId,
                        .title = MakeDetachedSurfaceWindowTitle(blockName, blockData.title),
                        .logicalSize = {
                            static_cast<F32>(surfaceMeta.detachedWidth),
                            static_cast<F32>(surfaceMeta.detachedHeight),
                        },
                        .configFields = buildConfigFields(flowgraphId,
                                                          blockName,
                                                          windowId,
                                                          blockData),
                        .configOpen = surfaceMeta.detachedConfigOpen,
                        .onResolveTexture = [texture]() {
                            return texture ? texture->raw() : 0;
                        },
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
                        .onMouse = [enqueue, surface](MouseEvent event) {
                            enqueue(MailSurfaceMouse{
                                .surface = surface,
                                .event = event,
                            });
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
                            enqueue(MailSetSurfaceConfigOpen{
                                .flowgraph = flowgraphId,
                                .block = blockName,
                                .surface = surfaceId,
                                .open = open,
                            });
                        },
                    });
                }
            }
        }

        return configs;
    }

 private:
    std::vector<FlowgraphConfigFieldConfig> buildConfigFields(const std::string& flowgraphId,
                                                              const std::string& blockName,
                                                              const std::string& windowId,
                                                              const Flowgraph::View::BlockData& blockData) const {
        auto fields = BuildFlowgraphConfigFields(windowId, blockData);
        for (auto& field : fields) {
            field.onApply = [enqueue = context.callbacks.enqueueMail, flowgraphId, blockName](Parser::Map patch, const bool silent) {
                enqueue(MailReconfigureBlock{flowgraphId,
                                             blockName,
                                             std::move(patch),
                                             silent});
            };
            field.onError = [enqueue = context.callbacks.enqueueMail](const Result result, const std::string& message) {
                enqueue(MailNotifyResult{.result = result, .message = message});
            };
            field.onBrowsePath = [enqueue = context.callbacks.enqueueMail](const bool save,
                                          std::vector<std::string> extensions,
                                          std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = "",
                    .save = save,
                    .extensions = std::move(extensions),
                    .onSelect = std::move(onSelect),
                });
            };
        }
        return fields;
    }

    std::unordered_set<std::string> buildReferencedSurfaceIds(const std::string& flowgraphId) const {
        std::unordered_set<std::string> referenced;
        const auto stacksIt = context.state.flowgraph.stacks.find(flowgraphId);
        if (stacksIt == context.state.flowgraph.stacks.end()) {
            return referenced;
        }

        for (const auto& [_, stack] : stacksIt->second) {
            if (!stack.meta.layout.has_value()) {
                continue;
            }
            collectReferencedSurfaceIds(flowgraphId, *stack.meta.layout, referenced);
        }
        return referenced;
    }

    static void collectReferencedSurfaceIds(const std::string& flowgraphId,
                                            const StackDockLayoutMeta& layout,
                                            std::unordered_set<std::string>& referenced) {
        if (layout.surfaces.has_value()) {
            for (const auto& surface : *layout.surfaces) {
                if (surface.block.empty() || surface.surface.empty()) {
                    continue;
                }
                referenced.insert(MakeDetachedSurfaceWindowId(flowgraphId, surface.block, surface.surface));
            }
        }
        if (!layout.children.has_value()) {
            return;
        }
        for (const auto& child : *layout.children) {
            collectReferencedSurfaceIds(flowgraphId, child, referenced);
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH
