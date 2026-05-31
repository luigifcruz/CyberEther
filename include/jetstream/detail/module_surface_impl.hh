#ifndef JETSTREAM_DETAIL_MODULE_SURFACE_IMPL_HH
#define JETSTREAM_DETAIL_MODULE_SURFACE_IMPL_HH

#include "../module_surface.hh"

#include <mutex>

namespace Jetstream {

struct Module::Surface::Impl {
    EventBuffer eventBuffer;
    std::vector<SurfaceManifest> manifests;

    mutable std::mutex eventMutex;
    mutable std::mutex manifestMutex;
};

}  // namespace Jetstream

#endif  // JETSTREAM_DETAIL_MODULE_SURFACE_IMPL_HH
