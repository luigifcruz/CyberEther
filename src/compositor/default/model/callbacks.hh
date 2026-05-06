#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_CALLBACKS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_CALLBACKS_HH

#include "messages.hh"

#include <functional>

namespace Jetstream {

struct DefaultCompositorCallbacks {
    std::function<void(Mail&&)> enqueueMail;
    std::function<void(std::function<Result()>, bool)> enqueueCommand;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_CALLBACKS_HH
