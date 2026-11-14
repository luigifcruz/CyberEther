#include "surface_callbacks.hh"

namespace Jetstream {

FlowgraphSurfaceCallbacks FlowgraphSurfaceCallbacks::Build(const Sakura::Callback<void(Mail&&)>& enqueue,
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

}  // namespace Jetstream
