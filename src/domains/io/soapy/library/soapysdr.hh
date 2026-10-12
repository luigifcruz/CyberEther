#ifndef JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_SOAPYSDR_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_LIBRARY_SOAPYSDR_HH

#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <jetstream/types.hh>

#include "../soapy_runtime.hh"

namespace Jetstream::Modules {

inline bool SoapyRangeContains(const std::vector<SoapyAbi::Range>& ranges, const F32 value) {
    for (const auto& range : ranges) {
        const F32 minimum = static_cast<F32>(range.minimum);
        const F32 maximum = static_cast<F32>(range.maximum);
        if (value < minimum || value > maximum) {
            continue;
        }

        const double step = range.step;
        if (!std::isfinite(step) || step <= 0.0) {
            return true;
        }

        const double stepCount = std::round((static_cast<double>(value) - range.minimum) / step);
        const F32 closest = static_cast<F32>(range.minimum + stepCount * step);
        if (value == closest) {
            return true;
        }
    }
    return false;
}

struct JETSTREAM_API SoapyDiscovery {
    using Clock = std::chrono::steady_clock;
    using DeviceEntry = SoapyKwargs;
    using DeviceList = std::map<std::string, DeviceEntry>;

    // Empty when SoapySDR is unavailable.
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

    // Saved device and stream strings stay opaque until SoapySDR parses them.
    Result open(const std::string& deviceString);
    Result openArgs(const SoapyKwargs& args);
    Result validateSettings(F32 sampleRate, F32 frequency) const;
    const std::vector<std::string>& listAntennas() const;
    Result setAntenna(const std::string& antenna);
    Result setTunerFrequency(F32 frequency);
    Result setSampleRate(F32 sampleRate);
    Result setAutomaticGain(bool automaticGain);
    Result setBiasTee(bool enabled);
    Result startStream(const std::string& streamString);
    Result startStreamArgs(const SoapyKwargs& args);
    void reset();

    ReadResult read(std::span<CF32> samples);

 private:
    enum class State {
        Closed,
        Open,
        StreamReady,
        Streaming,
    };

    Result queryCapabilities();
    std::string failure() const;

    State state = State::Closed;
    std::shared_ptr<SoapyLibrary> library;
    SoapyAbi::Device* device = nullptr;
    SoapyAbi::Stream* stream = nullptr;
    std::vector<SoapyAbi::Range> sampleRateRanges;
    std::vector<SoapyAbi::Range> frequencyRanges;
    std::vector<std::string> antennas;
    bool biasTeeSupported = false;
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
