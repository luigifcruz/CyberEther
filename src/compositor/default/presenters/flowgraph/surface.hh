#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH

#include "labels.hh"
#include "layout.hh"
#include "config.hh"
#include "surface_callbacks.hh"

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
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedSurfacePresenter {
    const PresenterContext& context;

    explicit FlowgraphDetachedSurfacePresenter(const PresenterContext& context) : context(context) {}

    std::vector<FlowgraphDetachedSurface::Config> build(const std::string& flowgraphId,
                                                        const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
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

                    if (!surfaceMeta.detached) {
                        continue;
                    }

                    const std::string key = MakeSurfaceTileKey(blockName, manifest.id);
                    auto callbacks = FlowgraphSurfaceCallbacks::Build(enqueue,
                                                                      surface,
                                                                      flowgraphId,
                                                                      blockName,
                                                                      surfaceMetaKey,
                                                                      SurfacePlacement::Detached);

                    configs.push_back({
                        .id = key,
                        .title = MakeSurfaceTileTitle(blockName, blockData.title),
                        .configFields = BuildFlowgraphDetachedConfigFields(enqueue,
                                                                           context.state.flowgraph,
                                                                           flowgraphId,
                                                                           blockName,
                                                                           key,
                                                                           blockData),
                        .configOpen = !surfaceMeta.configCollapsed,
                        .texture = manifest.surface,
                        .onSize = std::move(callbacks.onSize),
                        .onInput = std::move(callbacks.onInput),
                        .resolveCursor = std::move(callbacks.resolveCursor),
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

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH
