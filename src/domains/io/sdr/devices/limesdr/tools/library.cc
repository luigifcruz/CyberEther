#include "library.hh"

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <string_view>

#include <ConnectionRegistry.h>

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

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

std::vector<LmsApi::Board> BundledBoards() {
    static std::once_flag logging;
    std::call_once(logging, [] { LMS_RegisterLogHandler(&LogMessage); });

    std::vector<LmsApi::Board> boards;
    try {
        for (const auto& connection : lime::ConnectionRegistry::findConnections()) {
            boards.push_back({connection.name, connection.serial, connection.serialize()});
        }
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_LIMESDR] Failed to enumerate devices: {}", e.what());
    } catch (...) {
        JST_ERROR("[SDR_LIMESDR] Failed to enumerate devices.");
    }
    return boards;
}

//
// LimeSuiteNG
//

// LimeSuiteNG widens lms_name_t from 16 to 64 characters.
constexpr std::size_t NgNameLength = 64;
constexpr const char* NgLibraryOverride = "CYBERETHER_LIMESUITENG_LIBRARY";

int (*NgGetDeviceList)(lms_info_str_t*) = nullptr;

// Parses "LimeSDR XTRX, media=PCIe, addr=/dev/limepcie0, serial=000000000e5dd86c".
bool ParseNgBoard(const std::string& text, LmsApi::Board& board, std::string& media) {
    board = {.address = text};
    media.clear();

    std::string_view rest = text;
    bool first = true;
    while (!rest.empty()) {
        const auto comma = rest.find(", ");
        const auto field = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 2);

        const auto equals = field.find('=');
        if (equals == std::string_view::npos) {
            if (first) {
                board.name = field;
            }
        } else if (field.substr(0, equals) == "serial") {
            board.serial = field.substr(equals + 1);
        } else if (field.substr(0, equals) == "media") {
            media = field.substr(equals + 1);
        }
        first = false;
    }
    return !board.serial.empty();
}

// Lists PCIe boards only. USB boards are already served by the bundled
// LimeSuite, and listing them twice would offer two drivers for one radio.
std::vector<LmsApi::Board> NgBoards() {
    std::vector<LmsApi::Board> boards;
    const int count = NgGetDeviceList(nullptr);
    if (count <= 0) {
        return boards;
    }

    std::vector<lms_info_str_t> list(static_cast<std::size_t>(count));
    const int listed = NgGetDeviceList(list.data());
    for (int i = 0; i < std::min(listed, count); ++i) {
        const std::string text(list[i], strnlen(list[i], sizeof(lms_info_str_t)));
        LmsApi::Board board;
        std::string media;
        if (ParseNgBoard(text, board, media) && media == "PCIe") {
            boards.push_back(std::move(board));
        }
    }
    return boards;
}

template<typename F>
bool Bind(void* handle, const char* name, F& target) {
    std::string error;
    target = reinterpret_cast<F>(Platform::LoadDynamicLibrarySymbol(handle, name, error));
    return target != nullptr;
}

bool LoadNg(LmsApi& api) {
    std::vector<std::string> candidates;
    if (const char* override = std::getenv(NgLibraryOverride); override && *override) {
        candidates = {override};
    } else {
        for (const char* directory : {"", "/usr/local/lib/"}) {
            for (const char* name : {"liblimesuiteng-legacyapi.so.1.0.0", "liblimesuiteng-legacyapi.so.1",
                                     "liblimesuiteng-legacyapi.so"}) {
                candidates.push_back(std::string(directory) + name);
            }
        }
    }

    void* handle = nullptr;
    std::string error;
    for (const auto& candidate : candidates) {
        handle = Platform::OpenDynamicLibrary(candidate, Platform::DynamicLibraryVisibility::Local, error);
        if (handle) {
            JST_DEBUG("[SDR_LIMESDR] Loaded LimeSuiteNG from '{}'.", candidate);
            break;
        }
    }
    if (!handle) {
        JST_TRACE("[SDR_LIMESDR] LimeSuiteNG is not available: {}", error);
        return false;
    }

    void (*registerLogHandler)(LMS_LogHandler) = nullptr;
    const bool bound =
        Bind(handle, "LMS_GetDeviceList", NgGetDeviceList) &&
        Bind(handle, "LMS_RegisterLogHandler", registerLogHandler) &&
        Bind(handle, "LMS_Open", api.Open) &&
        Bind(handle, "LMS_Close", api.Close) &&
        Bind(handle, "LMS_Init", api.Init) &&
        Bind(handle, "LMS_EnableChannel", api.EnableChannel) &&
        Bind(handle, "LMS_SetSampleRate", api.SetSampleRate) &&
        Bind(handle, "LMS_GetSampleRate", api.GetSampleRate) &&
        Bind(handle, "LMS_GetSampleRateRange", api.GetSampleRateRange) &&
        Bind(handle, "LMS_SetLOFrequency", api.SetLOFrequency) &&
        Bind(handle, "LMS_GetLOFrequency", api.GetLOFrequency) &&
        Bind(handle, "LMS_GetLOFrequencyRange", api.GetLOFrequencyRange) &&
        Bind(handle, "LMS_GetAntennaList", api.GetAntennaList) &&
        Bind(handle, "LMS_SetAntenna", api.SetAntenna) &&
        Bind(handle, "LMS_SetGaindB", api.SetGaindB) &&
        Bind(handle, "LMS_SetLPFBW", api.SetLPFBW) &&
        Bind(handle, "LMS_GetLPFBWRange", api.GetLPFBWRange) &&
        Bind(handle, "LMS_Calibrate", api.Calibrate) &&
        Bind(handle, "LMS_EnableCache", api.EnableCache) &&
        Bind(handle, "LMS_SetupStream", api.SetupStream) &&
        Bind(handle, "LMS_DestroyStream", api.DestroyStream) &&
        Bind(handle, "LMS_StartStream", api.StartStream) &&
        Bind(handle, "LMS_StopStream", api.StopStream) &&
        Bind(handle, "LMS_RecvStream", api.RecvStream) &&
        Bind(handle, "LMS_GetStreamStatus", api.GetStreamStatus) &&
        Bind(handle, "LMS_GetLastErrorMessage", api.GetLastErrorMessage);
    if (!bound) {
        JST_WARN("[SDR_LIMESDR] LimeSuiteNG is missing part of the LimeSuite C API.");
        Platform::CloseDynamicLibrary(handle);
        return false;
    }

    // The library stays loaded for the life of the process.
    registerLogHandler(&LogMessage);
    api.boards = &NgBoards;
    api.nameLength = NgNameLength;
    return true;
}

}  // namespace

std::vector<std::pair<int, std::string>> LmsApi::antennas(lms_device_t* handle) const {
    std::vector<std::pair<int, std::string>> result;
    const int count = GetAntennaList(handle, LMS_CH_RX, 0, nullptr);
    if (count <= 0) {
        return result;
    }

    std::vector<char> names(static_cast<std::size_t>(count) * nameLength, '\0');
    if (GetAntennaList(handle, LMS_CH_RX, 0, reinterpret_cast<lms_name_t*>(names.data())) < 0) {
        return result;
    }
    for (int i = 0; i < count; ++i) {
        const char* name = names.data() + static_cast<std::size_t>(i) * nameLength;
        result.emplace_back(i, std::string(name, strnlen(name, nameLength)));
    }
    return result;
}

LmsApi& LmsApi::Bundled() {
    static LmsApi api;
    static std::once_flag flag;
    std::call_once(flag, [] {
        api.boards = &BundledBoards;
        api.Open = &LMS_Open;
        api.Close = &LMS_Close;
        api.Init = &LMS_Init;
        api.EnableChannel = &LMS_EnableChannel;
        api.SetSampleRate = &LMS_SetSampleRate;
        api.GetSampleRate = &LMS_GetSampleRate;
        api.GetSampleRateRange = &LMS_GetSampleRateRange;
        api.SetLOFrequency = &LMS_SetLOFrequency;
        api.GetLOFrequency = &LMS_GetLOFrequency;
        api.GetLOFrequencyRange = &LMS_GetLOFrequencyRange;
        api.GetAntennaList = &LMS_GetAntennaList;
        api.SetAntenna = &LMS_SetAntenna;
        api.SetGaindB = &LMS_SetGaindB;
        api.SetLPFBW = &LMS_SetLPFBW;
        api.GetLPFBWRange = &LMS_GetLPFBWRange;
        api.Calibrate = &LMS_Calibrate;
        api.EnableCache = &LMS_EnableCache;
        api.SetupStream = &LMS_SetupStream;
        api.DestroyStream = &LMS_DestroyStream;
        api.StartStream = &LMS_StartStream;
        api.StopStream = &LMS_StopStream;
        api.RecvStream = &LMS_RecvStream;
        api.GetStreamStatus = &LMS_GetStreamStatus;
        api.GetLastErrorMessage = &LMS_GetLastErrorMessage;
    });
    return api;
}

LmsApi* LmsApi::Ng() {
#if defined(JST_OS_LINUX)
    static LmsApi api;
    static const bool loaded = LoadNg(api);
    return loaded ? &api : nullptr;
#else
    return nullptr;
#endif
}

}  // namespace Jetstream::Radio
