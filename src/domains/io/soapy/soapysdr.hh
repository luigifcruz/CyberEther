#ifndef JETSTREAM_DOMAINS_IO_SOAPY_SOAPYSDR_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_SOAPYSDR_HH

#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>

#include <jetstream/types.hh>

namespace Jetstream::Modules {

inline bool SoapyRangeContains(const std::vector<SoapySDR::Range>& ranges, const F32 value) {
    for (const auto& range : ranges) {
        const F32 minimum = static_cast<F32>(range.minimum());
        const F32 maximum = static_cast<F32>(range.maximum());
        if (value < minimum || value > maximum) {
            continue;
        }

        const double step = range.step();
        if (!std::isfinite(step) || step <= 0.0) {
            return true;
        }

        const double stepCount = std::round((static_cast<double>(value) - range.minimum()) / step);
        const F32 closest = static_cast<F32>(range.minimum() + stepCount * step);
        if (value == closest) {
            return true;
        }
    }
    return false;
}

class SoapyReceiver {
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

    Result open(const SoapySDR::Kwargs& args);
    Result validateSettings(F32 sampleRate, F32 frequency) const;
    Result setTunerFrequency(F32 frequency);
    Result setSampleRate(F32 sampleRate);
    Result setAutomaticGain(bool automaticGain);
    Result setBiasTee(bool enabled);
    Result startStream(const SoapySDR::Kwargs& args);
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

    State state = State::Closed;
    SoapySDR::Device* device = nullptr;
    SoapySDR::Stream* stream = nullptr;
    std::vector<SoapySDR::Range> sampleRateRanges;
    std::vector<SoapySDR::Range> frequencyRanges;
    bool biasTeeSupported = false;
    bool biasTeeNeedsCleanup = false;
};

struct SoapyReceiveStatus {
    using Clock = std::chrono::steady_clock;

    enum class Action {
        Samples,
        Retry,
        WarnOverflow,
        Fail,
    };

    Action handle(SoapyReceiver::ReadStatus status,
                  Clock::time_point now = Clock::now());

    U64 deviceOverflows = 0;
    Clock::time_point nextWarning{};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_SOAPYSDR_HH
