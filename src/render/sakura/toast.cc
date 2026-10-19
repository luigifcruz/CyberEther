#include <jetstream/render/sakura/toast.hh>

#include "context.hh"
#include "metrics.hh"
#include "toast_queue.hh"

#include <mutex>

#include <regex>

namespace Jetstream::Sakura {

std::string CleanUserMessage(std::string message) {
    static const std::regex prefix(R"(^\[[^\]]+\]\s*)");
    return std::regex_replace(message, prefix, "");
}

namespace {

std::mutex& ToastMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<Private::QueuedToast>& ToastQueue() {
    static std::vector<Private::QueuedToast> queue;
    return queue;
}

}  // namespace

std::vector<Private::QueuedToast> Private::TakeToasts() {
    std::lock_guard<std::mutex> lock(ToastMutex());
    std::vector<QueuedToast> taken;
    taken.swap(ToastQueue());
    return taken;
}

void PushToast(const ToastType type, const I32 durationMs, const std::string& message, const std::string& title) {
    std::lock_guard<std::mutex> lock(ToastMutex());
    ToastQueue().push_back({.type = type, .durationMs = durationMs, .message = message, .title = title});
}

void PushToastResult(const Result value, const std::string& message) {
    const auto resolveMessage = [&](const std::string& fallback) {
        return CleanUserMessage(message.empty() ? fallback : message);
    };

    if (value == Result::ERROR) {
        PushToast(ToastType::Error, 5000, resolveMessage(JST_LOG_LAST_ERROR()));
    } else if (value == Result::FATAL) {
        PushToast(ToastType::Error, 5000, resolveMessage(JST_LOG_LAST_FATAL()));
    } else if (value == Result::WARNING) {
        PushToast(ToastType::Warning, 5000, resolveMessage(JST_LOG_LAST_WARNING()));
    } else if (value == Result::SUCCESS) {
        PushToast(ToastType::Success, 1000, "");
    } else if (value == Result::INCOMPLETE) {
        PushToast(ToastType::Info, 1000, resolveMessage(JST_LOG_LAST_ERROR()));
    }
}

}  // namespace Jetstream::Sakura
