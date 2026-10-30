#include "library.hh"

#include <string_view>

#include <LimeSuite.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

void LogMessage(const int level, const char* message) {
    std::string_view text = message ? message : "";
    while (text.ends_with('\n') || text.ends_with('\r')) {
        text.remove_suffix(1);
    }
    switch (level) {
        case LMS_LOG_CRITICAL:
        case LMS_LOG_ERROR:
            JST_ERROR("[SDR_LIMESDR] {}", text);
            break;
        case LMS_LOG_WARNING:
            JST_WARN("[SDR_LIMESDR] {}", text);
            break;
        case LMS_LOG_INFO:
            JST_DEBUG("[SDR_LIMESDR] {}", text);
            break;
        default:
            JST_TRACE("[SDR_LIMESDR] {}", text);
            break;
    }
}

}  // namespace

std::vector<lime::ConnectionHandle> LimeSdrLibrary::Boards() {
    static std::once_flag logging;
    std::call_once(logging, [] { LMS_RegisterLogHandler(&LogMessage); });
    try {
        return lime::ConnectionRegistry::findConnections();
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
