#ifndef JETSTREAM_RENDER_COMPONENTS_GENERIC_HH
#define JETSTREAM_RENDER_COMPONENTS_GENERIC_HH

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Render { class Window; }

namespace Jetstream::Render::Components {

class Generic {
 public:
    virtual ~Generic() = default;

    virtual Result create(Window*) {
        return Result::SUCCESS;
    }

    virtual Result destroy(Window*) {
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream::Render::Components

#endif