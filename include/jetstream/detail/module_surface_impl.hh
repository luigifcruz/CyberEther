#ifndef JETSTREAM_DETAIL_MODULE_SURFACE_IMPL_HH
#define JETSTREAM_DETAIL_MODULE_SURFACE_IMPL_HH

#include "../module_surface.hh"

namespace Jetstream {

struct Module::Surface::Impl {
    EventBuffer eventBuffer;
    std::vector<SurfaceManifest> manifests;
};

}  // namespace Jetstream

#endif  // JETSTREAM_DETAIL_MODULE_SURFACE_IMPL_HH
