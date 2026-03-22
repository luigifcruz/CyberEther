#ifndef JETSTREAM_MODULE_SURFACE_HH
#define JETSTREAM_MODULE_SURFACE_HH

#include <memory>

#include "jetstream/module.hh"
#include "jetstream/surface.hh"

namespace Jetstream {

struct Module::Surface {
 public:
    Surface();
    ~Surface();

    const std::vector<SurfaceManifest>& manifests() const;

    void pushMouseEvent(const MouseEvent& event);
    void pushSurfaceEvent(const SurfaceEvent& event);

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend struct Module::Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MODULE_SURFACE_HH
