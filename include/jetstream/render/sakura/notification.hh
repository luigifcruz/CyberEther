#pragma once

#include <jetstream/types.hh>

#include <string>

namespace Jetstream::Sakura {

enum class NotificationType {
    Info,
    Success,
    Warning,
    Error,
};

std::string CleanUserMessage(std::string message);
void Notify(NotificationType type, I32 durationMs, const std::string& message);
void NotifyResultClean(Result value, const std::string& message = "");

}  // namespace Jetstream::Sakura
