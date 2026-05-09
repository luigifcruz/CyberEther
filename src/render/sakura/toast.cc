#include <jetstream/render/sakura/toast.hh>

#include "base.hh"

namespace Jetstream::Sakura {

std::string CleanUserMessage(std::string message) {
    static const std::regex prefix(R"(^\[[^\]]+\]\s*)");
    return std::regex_replace(message, prefix, "");
}

void PushToast(const ToastType type, const I32 durationMs, const std::string& message) {
    ImGuiToastType toastType = ImGuiToastType_Info;
    switch (type) {
        case ToastType::Info:
            toastType = ImGuiToastType_Info;
            break;
        case ToastType::Success:
            toastType = ImGuiToastType_Success;
            break;
        case ToastType::Warning:
            toastType = ImGuiToastType_Warning;
            break;
        case ToastType::Error:
            toastType = ImGuiToastType_Error;
            break;
    }
    ImGui::InsertNotification({toastType, durationMs, message.c_str()});
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
