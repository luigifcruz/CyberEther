#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/node.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"

#include <memory>
#include <optional>
#include <string>

namespace Jetstream {

struct FlowgraphSurfacePresenter {
    const PresenterContext& context;

    explicit FlowgraphSurfacePresenter(const PresenterContext& context) : context(context) {}

    void buildSurfaces(FlowgraphNode::BlockData& block,
                       const std::shared_ptr<Flowgraph>& flowgraph,
                       const std::string& flowgraphId,
                       const std::string& blockName,
                       const std::string& nodeViewId,
                       const std::shared_ptr<Block>& blockPtr) const {
        const auto enqueue = context.callbacks.enqueueMail;
        for (const auto& surface : blockPtr->surfaces()) {
            for (const auto& manifest : surface->manifests()) {
                if (!manifest.surface || manifest.surface->raw() == 0) {
                    continue;
                }

                const std::string surfaceMetaKey = "surface_" + manifest.id;
                SurfaceMeta surfaceMeta;
                flowgraph->getMeta(surfaceMetaKey, surfaceMeta, blockName);
                std::optional<Extent2D<F32>> aspectRatioSize;
                const SurfaceMeta defaultSurfaceMeta;
                if (surfaceMeta.attachedWidth == defaultSurfaceMeta.attachedWidth &&
                    surfaceMeta.attachedHeight == defaultSurfaceMeta.attachedHeight &&
                    surfaceMeta.attachedWidth > 0 && surfaceMeta.attachedHeight > 0) {
                    aspectRatioSize = Extent2D<F32>{
                        static_cast<F32>(surfaceMeta.attachedWidth),
                        static_cast<F32>(surfaceMeta.attachedHeight),
                    };
                }

                block.surfaces.push_back({
                    .id = nodeViewId + ":surface:" + manifest.id,
                    .texture = manifest.surface,
                    .logicalSize = {
                        static_cast<F32>(surfaceMeta.detachedWidth),
                        static_cast<F32>(surfaceMeta.detachedHeight),
                    },
                    .aspectRatioSize = aspectRatioSize,
                    .detached = surfaceMeta.detached,
                    .onDetach = [enqueue, flowgraphId, blockName, surfaceId = manifest.id]() {
                        enqueue(MailSetSurfaceDetached{
                            .flowgraph = flowgraphId,
                            .block = blockName,
                            .surface = surfaceId,
                            .detached = true,
                        });
                    },
                    .onAttachedSize = [enqueue,
                                        surface,
                                       flowgraphId,
                                       surfaceMetaKey,
                                       blockName](const Sakura::SurfaceResize& resize) {
                        enqueue(MailResizeSurface{
                            .surface = surface,
                            .flowgraph = flowgraphId,
                            .block = blockName,
                            .metaKey = surfaceMetaKey,
                            .placement = SurfacePlacement::Attached,
                            .resize = {
                                .logicalSize = resize.logicalSize,
                                .framebufferSize = resize.framebufferSize,
                                .scale = resize.scale,
                            },
                        });
                    },
                    .onDetachedSize = [enqueue,
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
                });
            }
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH
