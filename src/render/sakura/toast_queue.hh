#ifndef JETSTREAM_RENDER_SAKURA_TOAST_QUEUE_HH
#define JETSTREAM_RENDER_SAKURA_TOAST_QUEUE_HH

#include <jetstream/render/sakura/toast.hh>

#include <string>
#include <vector>

namespace Jetstream::Sakura::Private {

struct QueuedToast {
    ToastType type = ToastType::Info;
    I32 durationMs = 3000;
    std::string message;
    std::string title;
};

std::vector<QueuedToast> TakeToasts();

}  // namespace Jetstream::Sakura::Private

#endif  // JETSTREAM_RENDER_SAKURA_TOAST_QUEUE_HH
