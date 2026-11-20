#include "library.hh"

#include <string_view>

#include <limesuiteng/DeviceRegistry.h>
#include <limesuiteng/Logger.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

void LogMessage(const lime::LogLevel level, const char* message) {
    std::string_view text = message ? message : "";
    while (text.ends_with('\n') || text.ends_with('\r')) {
        text.remove_suffix(1);
    }
    switch (level) {
        case lime::LogLevel::Critical:
        case lime::LogLevel::Error:
            JST_ERROR("[SDR_LIMESDR] {}", text);
            break;
        case lime::LogLevel::Warning:
            JST_WARN("[SDR_LIMESDR] {}", text);
            break;
        case lime::LogLevel::Info:
            JST_DEBUG("[SDR_LIMESDR] {}", text);
            break;
        default:
            JST_TRACE("[SDR_LIMESDR] {}", text);
            break;
    }
}

}  // namespace

std::vector<lime::DeviceHandle> LimeSdrLibrary::Boards() {
    static std::once_flag logging;
    std::call_once(logging, [] { lime::registerLogHandler(&LogMessage); });
    try {
        return lime::DeviceRegistry::enumerate();
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_LIMESDR] Failed to enumerate devices: {}", e.what());
    } catch (...) {
        JST_ERROR("[SDR_LIMESDR] Failed to enumerate devices.");
    }
    return {};
}

std::atomic<U64>& LimeSdrLibrary::OpenDevices() {
    static std::atomic<U64> count{0};
    return count;
}

std::mutex& LimeSdrLibrary::TuningMutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace Jetstream::Radio
