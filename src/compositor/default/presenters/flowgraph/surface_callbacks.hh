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
                                           const SurfacePlacement placement);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_CALLBACKS_HH
