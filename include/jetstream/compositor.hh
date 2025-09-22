#ifndef JETSTREAM_COMPOSITOR_HH
#define JETSTREAM_COMPOSITOR_HH

#include "jetstream/types.hh"
#include "jetstream/block.hh"

namespace Jetstream {

enum class JETSTREAM_API CompositorType {
    NONE = 0,
    DEFAULT,
};

class JETSTREAM_API Compositor {
 public:
    struct Impl;

    Compositor(const CompositorType& type);

    Result create(const std::shared_ptr<Instance>& instance,
                  const std::shared_ptr<Render::Window>& render,
                  const std::shared_ptr<Viewport::Generic>& viewport);
    Result destroy();

    Result present();
    Result poll();

 private:
    std::shared_ptr<Impl> impl;
};

}  // namespace Jetstream

#endif
