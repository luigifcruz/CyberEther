#include <jetstream/render/sakura/notification.hh>

#include "base.hh"

namespace Jetstream::Sakura {

std::string CleanUserMessage(std::string message) {
    static const std::regex prefix(R"(^\[[^\]]+\]\s*)");
    return std::regex_replace(message, prefix, "");
}

void Notify(const NotificationType type, const I32 durationMs, const std::string& message) {
    ImGuiToastType toastType = ImGuiToastType_Info;
    switch (type) {
        case NotificationType::Info:
            toastType = ImGuiToastType_Info;
            break;
        case NotificationType::Success:
            toastType = ImGuiToastType_Success;
            break;
        case NotificationType::Warning:
            toastType = ImGuiToastType_Warning;
            break;
        case NotificationType::Error:
            toastType = ImGuiToastType_Error;
            break;
    }
    ImGui::InsertNotification({toastType, durationMs, message.c_str()});
}

void NotifyResultClean(const Result value, const std::string& message) {
    const auto resolveMessage = [&](const std::string& fallback) {
        return CleanUserMessage(message.empty() ? fallback : message);
    };

    if (value == Result::ERROR) {
        Notify(NotificationType::Error, 5000, resolveMessage(JST_LOG_LAST_ERROR()));
    } else if (value == Result::FATAL) {
        Notify(NotificationType::Error, 5000, resolveMessage(JST_LOG_LAST_FATAL()));
    } else if (value == Result::WARNING) {
        Notify(NotificationType::Warning, 5000, resolveMessage(JST_LOG_LAST_WARNING()));
    } else if (value == Result::SUCCESS) {
        Notify(NotificationType::Success, 1000, "");
    } else if (value == Result::INCOMPLETE) {
        Notify(NotificationType::Info, 1000, resolveMessage(JST_LOG_LAST_ERROR()));
    }
}

}  // namespace Jetstream::Sakura
