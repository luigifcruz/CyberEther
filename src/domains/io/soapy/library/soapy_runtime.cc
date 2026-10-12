#include "soapy_runtime.hh"

#include <cstdlib>
#include <mutex>

#include <jetstream/logger.hh>
#include <jetstream/platform.hh>

namespace Jetstream::Modules {

namespace {

constexpr const char* LibraryOverride = "CYBERETHER_SOAPYSDR_LIBRARY";

struct RuntimeCache {
    std::mutex mutex;
    std::shared_ptr<SoapyLibrary> library;
    SoapyRuntime::Status status;
    bool attempted = false;
};

RuntimeCache& Cache() {
    static RuntimeCache cache;
    return cache;
}

template<typename T>
bool Resolve(void* handle, T& target, const char* symbol, std::string& missing) {
    std::string error;
    target = reinterpret_cast<T>(Platform::LoadDynamicLibrarySymbol(handle, symbol, error));
    if (!target) {
        missing = symbol;
        return false;
    }
    return true;
}

bool ResolveAll(void* handle, SoapyApi& api, std::string& missing) {
    return Resolve(handle, api.getABIVersion, "SoapySDR_getABIVersion", missing) &&
           Resolve(handle, api.getAPIVersion, "SoapySDR_getAPIVersion", missing) &&
           Resolve(handle, api.lastError, "SoapySDRDevice_lastError", missing) &&
           Resolve(handle, api.lastStatus, "SoapySDRDevice_lastStatus", missing) &&
           Resolve(handle, api.errToStr, "SoapySDR_errToStr", missing) &&
           Resolve(handle, api.free, "SoapySDR_free", missing) &&
           Resolve(handle, api.stringsClear, "SoapySDRStrings_clear", missing) &&
           Resolve(handle, api.kwargsFromString, "SoapySDRKwargs_fromString", missing) &&
           Resolve(handle, api.kwargsToString, "SoapySDRKwargs_toString", missing) &&
           Resolve(handle, api.kwargsClear, "SoapySDRKwargs_clear", missing) &&
           Resolve(handle, api.kwargsListClear, "SoapySDRKwargsList_clear", missing) &&
           Resolve(handle, api.argInfoListClear, "SoapySDRArgInfoList_clear", missing) &&
           Resolve(handle, api.loadModule, "SoapySDR_loadModule", missing) &&
           Resolve(handle, api.enumerate, "SoapySDRDevice_enumerate", missing) &&
           Resolve(handle, api.make, "SoapySDRDevice_make", missing) &&
           Resolve(handle, api.unmake, "SoapySDRDevice_unmake", missing) &&
           Resolve(handle, api.getSampleRateRange, "SoapySDRDevice_getSampleRateRange", missing) &&
           Resolve(handle, api.getFrequencyRange, "SoapySDRDevice_getFrequencyRange", missing) &&
           Resolve(handle, api.listAntennas, "SoapySDRDevice_listAntennas", missing) &&
           Resolve(handle, api.setAntenna, "SoapySDRDevice_setAntenna", missing) &&
           Resolve(handle, api.setFrequency, "SoapySDRDevice_setFrequency", missing) &&
           Resolve(handle, api.setSampleRate, "SoapySDRDevice_setSampleRate", missing) &&
           Resolve(handle, api.setGainMode, "SoapySDRDevice_setGainMode", missing) &&
           Resolve(handle, api.getGainRange, "SoapySDRDevice_getGainRange", missing) &&
           Resolve(handle, api.setGain, "SoapySDRDevice_setGain", missing) &&
           Resolve(handle, api.getSettingInfo, "SoapySDRDevice_getSettingInfo", missing) &&
           Resolve(handle, api.writeSetting, "SoapySDRDevice_writeSetting", missing) &&
           Resolve(handle, api.setupStream, "SoapySDRDevice_setupStream", missing) &&
           Resolve(handle, api.closeStream, "SoapySDRDevice_closeStream", missing) &&
           Resolve(handle, api.activateStream, "SoapySDRDevice_activateStream", missing) &&
           Resolve(handle, api.deactivateStream, "SoapySDRDevice_deactivateStream", missing) &&
           Resolve(handle, api.readStream, "SoapySDRDevice_readStream", missing);
}

SoapyKwargs ToMap(const SoapyAbi::Kwargs& kwargs) {
    SoapyKwargs map;
    for (std::size_t i = 0; i < kwargs.size; ++i) {
        map[kwargs.keys[i] ? kwargs.keys[i] : ""] = kwargs.vals[i] ? kwargs.vals[i] : "";
    }
    return map;
}

}  // namespace

SoapyLibrary::~SoapyLibrary() {
    Platform::CloseDynamicLibrary(handle);
}

const SoapyApi& SoapyLibrary::api() const {
    return table;
}

const std::string& SoapyLibrary::path() const {
    return libraryPath;
}

std::string SoapyLibrary::lastError() const {
    const char* error = table.lastError();
    return error ? error : "";
}

std::string SoapyLibrary::errorString(const int code) const {
    const char* text = table.errToStr(code);
    return text ? text : "UNKNOWN";
}

Result SoapyLibrary::kwargsFromString(const std::string& markup, SoapyKwargs& kwargs) const {
    auto parsed = table.kwargsFromString(markup.c_str());
    kwargs = ToMap(parsed);
    table.kwargsClear(&parsed);
    return Result::SUCCESS;
}

std::string SoapyLibrary::kwargsToString(const SoapyKwargs& kwargs) const {
    const SoapyKwargsView view(kwargs);
    char* markup = table.kwargsToString(view.get());
    std::string result = markup ? markup : "";
    table.free(markup);
    return result;
}

Result SoapyLibrary::enumerate(const SoapyKwargs& filter, SoapyKwargsList& devices) const {
    devices.clear();
    const SoapyKwargsView view(filter);
    std::size_t length = 0;
    auto* list = table.enumerate(view.get(), &length);
    if (table.lastStatus() != 0) {
        table.kwargsListClear(list, length);
        JST_ERROR("[MODULE_SOAPY] Failed to enumerate devices: {}", lastError());
        return Result::ERROR;
    }
    for (std::size_t i = 0; i < length; ++i) {
        devices.push_back(ToMap(list[i]));
    }
    table.kwargsListClear(list, length);
    return Result::SUCCESS;
}

Result SoapyLibrary::loadModule(const std::string& path) const {
    char* error = table.loadModule(path.c_str());
    const std::string message = error ? error : "";
    table.free(error);
    if (message.empty() || message.ends_with(" already loaded")) {
        return Result::SUCCESS;
    }
    JST_ERROR("[MODULE_SOAPY] Failed to load SoapySDR module '{}': {}", path, message);
    return Result::ERROR;
}

std::shared_ptr<SoapyLibrary> SoapyLibrary::Open(const std::string& path, std::string& error) {
    void* handle = Platform::OpenDynamicLibrary(path, Platform::DynamicLibraryVisibility::Global,
                                                error);
    if (!handle) {
        return nullptr;
    }

    std::shared_ptr<SoapyLibrary> library(new SoapyLibrary());
    library->handle = handle;
    library->libraryPath = path;

    std::string missing;
    if (!Resolve(handle, library->table.getABIVersion, "SoapySDR_getABIVersion", missing)) {
        error = "not a SoapySDR library (missing SoapySDR_getABIVersion)";
        return nullptr;
    }
    const char* abi = library->table.getABIVersion();
    const std::string version = abi ? abi : "";
    if (version != SoapyAbi::Version) {
        error = jst::fmt::format("ABI version '{}' is not the supported '{}'", version,
                                 SoapyAbi::Version);
        return nullptr;
    }
    if (!ResolveAll(handle, library->table, missing)) {
        error = jst::fmt::format("missing symbol {}", missing);
        return nullptr;
    }
    return library;
}

SoapyKwargsView::SoapyKwargsView(const SoapyKwargs& kwargs) {
    keys.reserve(kwargs.size());
    values.reserve(kwargs.size());
    for (const auto& [key, value] : kwargs) {
        keys.push_back(const_cast<char*>(key.c_str()));
        values.push_back(const_cast<char*>(value.c_str()));
    }
    view.size = kwargs.size();
    view.keys = keys.data();
    view.vals = values.data();
}

const SoapyAbi::Kwargs* SoapyKwargsView::get() const {
    return &view;
}

std::vector<std::string> SoapyRuntime::Candidates() {
    if (const char* override = std::getenv(LibraryOverride); override && *override) {
        return {override};
    }

#if defined(JST_OS_MAC)
    std::vector<std::string> names{"libSoapySDR.0.8.dylib", "libSoapySDR.dylib"};
    std::vector<std::string> directories{"", "/opt/homebrew/lib/", "/usr/local/lib/",
                                         "/opt/local/lib/"};
#elif defined(JST_OS_LINUX)
    std::vector<std::string> names{"libSoapySDR.so.0.8", "libSoapySDR.so"};
    std::vector<std::string> directories{"", "/usr/local/lib/"};
#elif defined(JST_OS_WINDOWS)
    std::vector<std::string> names{"SoapySDR.dll", "libSoapySDR.dll"};
    std::vector<std::string> directories{""};
#else
    std::vector<std::string> names;
    std::vector<std::string> directories;
#endif

    std::vector<std::string> candidates;
    for (const auto& name : names) {
        for (const auto& directory : directories) {
            candidates.push_back(directory + name);
        }
    }
    return candidates;
}

std::shared_ptr<SoapyLibrary> SoapyRuntime::Acquire() {
    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    if (cache.attempted) {
        return cache.library;
    }
    cache.attempted = true;
    cache.status = {};

#if defined(JST_OS_BROWSER) || defined(JST_OS_IOS) || defined(JST_OS_ANDROID)
    cache.status.message = "SoapySDR is not supported on this platform.";
    return nullptr;
#else
    for (const auto& candidate : Candidates()) {
        std::string error;
        auto library = SoapyLibrary::Open(candidate, error);
        if (!library) {
            cache.status.attempts.push_back(jst::fmt::format("{}: {}", candidate, error));
            continue;
        }
        cache.library = std::move(library);
        cache.status.available = true;
        cache.status.path = candidate;
        const char* version = cache.library->api().getAPIVersion();
        cache.status.version = version ? version : "";
        JST_DEBUG("[MODULE_SOAPY] Loaded SoapySDR {} from '{}'.", cache.status.version, candidate);
        return cache.library;
    }

    cache.status.message = "SoapySDR 0.8 was not found. Install SoapySDR and its device "
                           "modules from your package manager, or use the SDR block.";
    JST_DEBUG("[MODULE_SOAPY] {}", cache.status.message);
    for (const auto& attempt : cache.status.attempts) {
        JST_DEBUG("[MODULE_SOAPY] Tried {}", attempt);
    }
    return nullptr;
#endif
}

SoapyRuntime::Status SoapyRuntime::Availability() {
    Acquire();
    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    return cache.status;
}

void SoapyRuntime::Reset() {
    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    cache.library.reset();
    cache.status = {};
    cache.attempted = false;
}

}  // namespace Jetstream::Modules
