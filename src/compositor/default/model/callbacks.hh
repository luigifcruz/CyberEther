#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_CALLBACKS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_CALLBACKS_HH

#include "messages.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/types.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct DefaultCompositorCallbacks {
    std::function<void(Mail&&)> enqueueMail;
    std::function<void(std::function<Result()>, bool)> enqueueCommand;
    std::function<void(Sakura::ToastType, I32, const std::string&)> notify;
    std::function<void(Result, const std::string&)> notifyResult;
    std::function<void(const std::string&)> setClipboardText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_CALLBACKS_HH
