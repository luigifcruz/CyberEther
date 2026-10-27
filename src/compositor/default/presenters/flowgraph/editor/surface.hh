#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH

#include "../../context.hh"
#include "../surface_callbacks.hh"

#include "../../../model/messages.hh"
#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/node.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
#include <string>
#include <utility>

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

                auto callbacks = FlowgraphSurfaceCallbacks::Build(enqueue,
                                                                  surface,
                                                                  flowgraphId,
                                                                  blockName,
                                                                  surfaceMetaKey,
                                                                  SurfacePlacement::Attached);
                block.surfaces.push_back({
                    .id = jst::fmt::format("{}:surface:{}", nodeViewId, manifest.id),
                    .texture = manifest.surface,
                    .height = static_cast<F32>(surfaceMeta.attachedHeight),
                    .detached = surfaceMeta.detached,
                    .onDetach = [enqueue, flowgraphId, blockName, surfaceId = manifest.id](bool detached) {
                        enqueue(MailSetSurfaceDetached{
                            .flowgraph = flowgraphId,
                            .block = blockName,
                            .surface = surfaceId,
                            .detached = detached,
                        });
                    },
                    .onAttachedSize = std::move(callbacks.onSize),
                    .onInput = std::move(callbacks.onInput),
                    .resolveCursor = std::move(callbacks.resolveCursor),
                });
            }
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH
