#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_LIBRARY_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_LIBRARY_HH

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream::Radio::SoapyAbi {

constexpr std::array<const char*, 2> Versions = {"0.8", "0.9"};

constexpr int RX = 1;
constexpr int EndBurst = 1 << 1;
constexpr int HasTime = 1 << 2;
constexpr int EndAbrupt = 1 << 3;
constexpr int Timeout = -1;
constexpr int StreamError = -2;
constexpr int Overflow = -4;
constexpr int NotSupported = -5;
constexpr const char* FormatCF32 = "CF32";
constexpr const char* FormatCF64 = "CF64";
constexpr const char* FormatCS16 = "CS16";
constexpr const char* FormatCS8 = "CS8";
constexpr const char* FormatCU8 = "CU8";

struct Range {
    double minimum;
    double maximum;
    double step;
};

struct Kwargs {
    std::size_t size;
    char** keys;
    char** vals;
};

struct ArgInfo {
    char* key;
    char* value;
    char* name;
    char* description;
    char* units;
    int type;
    Range range;
    std::size_t numOptions;
    char** options;
    char** optionNames;
};

struct Device;
struct Stream;

}  // namespace Jetstream::Radio::SoapyAbi

namespace Jetstream::Radio {

using SoapyKwargs = std::map<std::string, std::string>;
using SoapyKwargsList = std::vector<SoapyKwargs>;

struct SoapyApi {
    const char* (*getABIVersion)() = nullptr;
    const char* (*getAPIVersion)() = nullptr;
    const char* (*lastError)() = nullptr;
    int (*lastStatus)() = nullptr;
    const char* (*errToStr)(int) = nullptr;
    void (*free)(void*) = nullptr;
    void (*stringsClear)(char***, std::size_t) = nullptr;
    SoapyAbi::Kwargs (*kwargsFromString)(const char*) = nullptr;
    char* (*kwargsToString)(const SoapyAbi::Kwargs*) = nullptr;
    void (*kwargsClear)(SoapyAbi::Kwargs*) = nullptr;
    void (*kwargsListClear)(SoapyAbi::Kwargs*, std::size_t) = nullptr;
    void (*argInfoListClear)(SoapyAbi::ArgInfo*, std::size_t) = nullptr;
    SoapyAbi::Kwargs* (*enumerate)(const SoapyAbi::Kwargs*, std::size_t*) = nullptr;
    SoapyAbi::Device* (*make)(const SoapyAbi::Kwargs*) = nullptr;
    int (*unmake)(SoapyAbi::Device*) = nullptr;
    std::size_t (*getNumChannels)(const SoapyAbi::Device*, int) = nullptr;
    SoapyAbi::Range* (*getFrequencyRange)(const SoapyAbi::Device*, int, std::size_t,
                                          std::size_t*) = nullptr;
    SoapyAbi::Range* (*getSampleRateRange)(const SoapyAbi::Device*, int, std::size_t,
                                           std::size_t*) = nullptr;
    char** (*listAntennas)(const SoapyAbi::Device*, int, std::size_t, std::size_t*) = nullptr;
    char* (*getNativeStreamFormat)(const SoapyAbi::Device*, int, std::size_t,
                                   double*) = nullptr;
    char** (*getStreamFormats)(const SoapyAbi::Device*, int, std::size_t, std::size_t*) = nullptr;
    int (*setAntenna)(SoapyAbi::Device*, int, std::size_t, const char*) = nullptr;
    char* (*getAntenna)(const SoapyAbi::Device*, int, std::size_t) = nullptr;
    int (*setFrequency)(SoapyAbi::Device*, int, std::size_t, double,
                        const SoapyAbi::Kwargs*) = nullptr;
    int (*setSampleRate)(SoapyAbi::Device*, int, std::size_t, double) = nullptr;
    double (*getSampleRate)(const SoapyAbi::Device*, int, std::size_t) = nullptr;
    int (*setBandwidth)(SoapyAbi::Device*, int, std::size_t, double) = nullptr;
    SoapyAbi::Range* (*getBandwidthRange)(const SoapyAbi::Device*, int, std::size_t,
                                          std::size_t*) = nullptr;
    double (*getFrequency)(const SoapyAbi::Device*, int, std::size_t) = nullptr;
    bool (*hasGainMode)(const SoapyAbi::Device*, int, std::size_t) = nullptr;
    int (*setGainMode)(SoapyAbi::Device*, int, std::size_t, bool) = nullptr;
    SoapyAbi::Range (*getGainRange)(const SoapyAbi::Device*, int, std::size_t) = nullptr;
    int (*setGain)(SoapyAbi::Device*, int, std::size_t, double) = nullptr;
    SoapyAbi::ArgInfo* (*getSettingInfo)(const SoapyAbi::Device*, std::size_t*) = nullptr;
    int (*writeSetting)(SoapyAbi::Device*, const char*, const char*) = nullptr;
    SoapyAbi::Stream* (*setupStream)(SoapyAbi::Device*, int, const char*, const std::size_t*,
                                     std::size_t, const SoapyAbi::Kwargs*) = nullptr;
    int (*closeStream)(SoapyAbi::Device*, SoapyAbi::Stream*) = nullptr;
    int (*activateStream)(SoapyAbi::Device*, SoapyAbi::Stream*, int, long long,
                          std::size_t) = nullptr;
    int (*deactivateStream)(SoapyAbi::Device*, SoapyAbi::Stream*, int, long long) = nullptr;
    int (*readStream)(SoapyAbi::Device*, SoapyAbi::Stream*, void* const*, std::size_t, int*,
                      long long*, long) = nullptr;
    bool (*hasHardwareTime)(const SoapyAbi::Device*, const char*) = nullptr;
    long long (*getHardwareTime)(const SoapyAbi::Device*, const char*) = nullptr;
};

class JETSTREAM_API SoapyLibrary {
 public:
    ~SoapyLibrary();

    const SoapyApi& api() const;
    const std::string& path() const;

    std::string lastError() const;
    std::string errorString(int code) const;

    Result kwargsFromString(const std::string& markup, SoapyKwargs& kwargs) const;
    std::string kwargsToString(const SoapyKwargs& kwargs) const;
    Result enumerate(const SoapyKwargs& filter, SoapyKwargsList& devices) const;

    static std::shared_ptr<SoapyLibrary> Open(const std::string& path, std::string& error);

 private:
    SoapyLibrary() = default;

    void* handle = nullptr;
    std::string libraryPath;
    SoapyApi table;

    friend struct SoapyRuntime;
};

class JETSTREAM_API SoapyKwargsView {
 public:
    explicit SoapyKwargsView(const SoapyKwargs& kwargs);

    const SoapyAbi::Kwargs* get() const;

 private:
    std::vector<char*> keys;
    std::vector<char*> values;
    SoapyAbi::Kwargs view{};
};

struct JETSTREAM_API SoapyRuntime {
    struct Status {
        bool available = false;
        std::string message;
        std::string path;
        std::string version;
        std::vector<std::string> attempts;
    };

    static std::shared_ptr<SoapyLibrary> Acquire();
    static Status Availability();

    static std::vector<std::string> Candidates();

    static void Reset();
    static void Override(const SoapyApi& api);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_LIBRARY_HH
