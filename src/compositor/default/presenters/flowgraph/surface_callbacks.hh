#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_CALLBACKS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_CALLBACKS_HH

#include "../../model/messages.hh"

#include "jetstream/render/sakura/surface.hh"
#include "jetstream/surface.hh"

#include <memory>
#include <string>

namespace Jetstream {

struct FlowgraphSurfaceCallbacks {
    Sakura::Callback<void(const Sakura::SurfaceResize&)> onSize;
    Sakura::Callback<void(InputEvent)> onInput;
    Sakura::Callback<SurfaceCursor()> resolveCursor;

    static FlowgraphSurfaceCallbacks Build(const Sakura::Callback<void(Mail&&)>& enqueue,
                                           const std::shared_ptr<Module::Surface>& surface,
                                           const std::string& flowgraphId,
                                           const std::string& blockName,
                                           const std::string& metaKey,
                                           const SurfacePlacement placement) {
        return {
            .onSize = [enqueue, surface, flowgraphId, blockName, metaKey, placement](
                          const Sakura::SurfaceResize& resize) {
                enqueue(MailResizeSurface{
                    .surface = surface,
                    .flowgraph = flowgraphId,
                    .block = blockName,
                    .metaKey = metaKey,
                    .placement = placement,
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
            .resolveCursor = [surface]() {
                return surface->cursor();
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_CALLBACKS_HH
