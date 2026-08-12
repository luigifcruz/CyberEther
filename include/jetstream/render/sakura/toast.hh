#ifndef JETSTREAM_RENDER_SAKURA_TOAST_HH
#define JETSTREAM_RENDER_SAKURA_TOAST_HH

#include <jetstream/types.hh>

#include <string>

namespace Jetstream::Sakura {

enum class ToastType {
    Info,
    Success,
    Warning,
    Error,
};

std::string CleanUserMessage(std::string message);
void PushToast(ToastType type, I32 durationMs, const std::string& message);
void PushToastResult(Result value, const std::string& message = "");

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TOAST_HH
