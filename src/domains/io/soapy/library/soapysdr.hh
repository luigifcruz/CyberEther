#ifndef JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_SOAPYSDR_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_SOAPYSDR_HH

#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <jetstream/types.hh>

#include "soapy_runtime.hh"
#include "formats.hh"

namespace Jetstream::Modules {

template<typename T>
struct SoapyQuery {
    std::optional<T> value;
    std::string error;
};

struct SoapyCapabilities {
    SoapyQuery<std::vector<SoapyAbi::Range>> sampleRates;
    SoapyQuery<std::vector<SoapyAbi::Range>> frequencies;
    SoapyQuery<std::vector<std::string>> antennas;
    SoapyQuery<std::vector<std::string>> dataTypes;
    SoapyQuery<SoapyAbi::Range> gain;
    SoapyQuery<bool> automaticGain;
    SoapyQuery<bool> biasTee;
};

struct SoapyDeviceState {
    SoapyQuery<F64> frequency;
    SoapyQuery<F64> sampleRate;
    SoapyQuery<std::string> antenna;
    SoapyQuery<bool> automaticGain;
    SoapyQuery<F64> gain;
};

struct SoapyDeviceInfo {
    bool connected = false;
    SoapyCapabilities capabilities;
    SoapyDeviceState state;
};

struct JETSTREAM_API SoapyDiscovery {
    using Clock = std::chrono::steady_clock;
    using DeviceEntry = SoapyKwargs;
    using DeviceList = std::map<std::string, DeviceEntry>;

    static DeviceList List(const std::string& filter = "", bool refresh = false,
                           std::optional<Clock::time_point> now = std::nullopt);
    static void ClearCache();
    static Result LoadDriverLibrary(const std::string& path);
    static DeviceList BuildDeviceList(const SoapyKwargsList& entries);
    static std::string FormatArguments(const SoapyKwargs& args);
};

class JETSTREAM_API SoapyReceiver {
 public:
    enum class ReadStatus {
        Samples,
        Timeout,
        Overflow,
        Error,
    };

    struct ReadResult {
        ReadStatus status = ReadStatus::Error;
        std::size_t sampleCount = 0;
        std::optional<I64> timestampNs;
        std::string error;
    };

    SoapyReceiver() = default;
    ~SoapyReceiver();
    SoapyReceiver(const SoapyReceiver&) = delete;
    SoapyReceiver& operator=(const SoapyReceiver&) = delete;

    Result open(const std::string& deviceString);
    Result openArgs(const SoapyKwargs& args);
    const SoapyDeviceInfo& info() const;
    Result refresh();
    Result validateDataType(const DataType& dtype) const;
    F32 clampGain(F32 gain) const;
    Result setAntenna(const std::string& antenna);
    Result setTunerFrequency(F32 frequency);
    Result setSampleRate(F32 sampleRate);
    Result setAutomaticGain(bool automaticGain);
    Result setGain(F32 gain);
    Result setBiasTee(bool enabled);
    Result startStream(const std::string& streamString, const DataType& dtype);
    Result startStreamArgs(const SoapyKwargs& args, const DataType& dtype);
    void reset();

    ReadResult read(void* samples, std::size_t count);
    F32 sampleScale() const;

 private:
    enum class State {
        Closed,
        Open,
        StreamReady,
        Streaming,
    };

    Result queryCapabilities();
    void queryTuningCapabilities();
    Result queryState();
    std::string failure() const;

    State state = State::Closed;
    std::shared_ptr<SoapyLibrary> library;
    SoapyAbi::Device* device = nullptr;
    SoapyAbi::Stream* stream = nullptr;
    SoapyDeviceInfo deviceInfo;
    F32 scale = 1.0f;
    bool biasTeeNeedsCleanup = false;
};

struct JETSTREAM_API SoapyReceiveStatus {
    enum class Action {
        Samples,
        Retry,
        Fail,
    };

    Action handle(SoapyReceiver::ReadStatus status);

    U64 deviceOverflows = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_SOAPYSDR_HH
