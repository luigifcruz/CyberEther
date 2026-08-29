#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/node.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
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
                       const Flowgraph::View::BlockData& blockData) const {
        const auto enqueue = context.callbacks.enqueueMail;
        for (const auto& surface : blockData.surfaces) {
            for (const auto& manifest : surface->manifests()) {
                if (!manifest.surface || manifest.surface->raw() == 0) {
                    continue;
                }

                const std::string surfaceMetaKey = "surface_" + manifest.id;
                SurfaceMeta surfaceMeta;
                flowgraph->metadata().get(surfaceMetaKey, surfaceMeta, blockName);

                block.surfaces.push_back({
                    .id = nodeViewId + ":surface:" + manifest.id,
                    .texture = manifest.surface,
                    .height = static_cast<F32>(surfaceMeta.attachedHeight),
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
