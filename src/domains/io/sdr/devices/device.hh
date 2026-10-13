#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_DEVICE_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_DEVICE_HH

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

struct JETSTREAM_API Device {
 public:
    enum class Status : U8 {
        Samples,
        Timeout,
        Discontinuity,
        Disconnected,
        Failure,
    };

    struct JETSTREAM_API Range {
        F64 minimum = 0.0;
        F64 maximum = 0.0;

        static bool Contains(const std::vector<Range>& ranges, F64 value);
    };

    struct SweepLayout {
        struct Band {
            U64 offset = 0;
            U64 length = 0;
        };

        struct Hop {
            F64 frequency = 0.0;
            std::vector<Band> bands;
        };

        F64 startFrequency = 0.0;
        F64 endFrequency = 0.0;
        std::vector<Hop> hops;
    };

    struct FirmwareSweep {
        using Layout = std::function<bool(F64 startFrequency, F64 endFrequency, U64 captureLength,
                                          SweepLayout& layout)>;

        F64 sampleRate = 0.0;
        U64 maxDwell = 0;
        U64 maxCaptureLength = 0;
        U64 captureBase = 1;
        U64 captureStep = 1;
        Layout layout;
    };

    struct JETSTREAM_API Info {
        std::string driver;
        std::string serial;
        std::string label;
        bool biasTee = true;

        std::string selector() const;
    };

    struct JETSTREAM_API Selection {
        std::string driver;
        std::string serial;

        bool matches(const Info& info) const;

        static std::optional<Selection> Parse(const std::string& selector);
    };

    struct JETSTREAM_API Capabilities {
        std::vector<Range> frequencyRanges;
        std::vector<Range> sampleRateRanges;
        std::vector<std::string> antennas;
        bool automaticGain = false;
        bool biasTee = false;
        F64 sweepSampleRate = 0.0;
        F64 usableBandwidth = 0.75;
        F64 dcExclusion = 0.0;
        std::optional<FirmwareSweep> firmwareSweep;

        std::vector<F64> fixedSampleRates() const;
    };

    struct Settings {
        F64 frequency = 96.9e6;
        F64 sampleRate = 2.0e6;
        std::string antenna;
        bool automaticGain = true;
        bool biasTee = false;
        F64 settlingTime = -1.0;
        F64 gain = 30.0;
        bool lowLatency = false;
    };

    struct Applied {
        F64 frequency = 0.0;
        F64 sampleRate = 0.0;
        std::string antenna;
        bool automaticGain = false;
        bool biasTee = false;
        F64 gain = 0.0;
        std::optional<Range> gainRange;
    };

    struct Statistics {
        U64 failedHops = 0;
    };

    struct Capture {
        Status status = Status::Failure;
        U64 count = 0;
        U64 generation = 0;
        F64 frequency = 0.0;
        F64 sampleRate = 0.0;
        std::optional<U64> timestamp;
        U64 epoch = 0;
        U64 lost = 0;
        std::string error;
    };

    struct SweepRequest {
        bool firmware = false;
        std::vector<F64> frequencies;
        F64 sampleRate = 0.0;
        U64 captureLength = 0;
        U64 dwell = 1;
        F64 startFrequency = 0.0;
        F64 endFrequency = 0.0;

        bool operator==(const SweepRequest&) const = default;
    };

    struct SweepCapture {
        U64 hop = 0;
        F64 frequency = 0.0;
        std::span<const CF32> samples;
    };

    using SweepSink = std::function<bool(const SweepCapture&)>;
    using SweepRunning = std::function<bool()>;

    struct Driver {
        std::string name;
        std::function<std::vector<Info>()> enumerate;
        std::function<Result(const Info& info, std::unique_ptr<Device>& device)> create;
    };

    using Labels = std::vector<std::pair<std::string, Info>>;

    virtual ~Device() = default;

    static std::vector<Driver> Drivers();
    static void Override(std::optional<std::vector<Driver>> drivers);
    static std::vector<Info> List(bool refresh = false);
    static void ClearCache();
    static Labels Label(const std::vector<Info>& devices);
    static std::optional<Capabilities> Cached(const std::string& selector);
    static bool SupportsBiasTee(const std::string& selector);
    static Result Open(const std::string& selector, std::unique_ptr<Device>& device);

    const Info& info() const;
    const Capabilities& capabilities() const;
    const Applied& applied() const;
    const Settings& settings() const;
    U64 generation() const;
    Statistics statistics() const;

    virtual Result start(const Settings& settings) = 0;
    virtual void stop() = 0;
    virtual Result apply(const Settings& settings);
    Result update(const Settings& settings);
    virtual Result tune(F64 frequency);
    virtual Result setAutomaticGain(bool automaticGain);
    virtual Result setGain(F64 gain);
    virtual F64 resolveGain(F64 gain, const std::optional<Range>& range) const;
    Result setSettlingTime(F64 settlingTime);
    virtual Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) = 0;
    virtual Result prepareSweep(const SweepRequest& request);
    virtual Result runSweep(const SweepRunning& running, const SweepSink& sink);

 protected:
    static constexpr std::chrono::seconds StallTimeout{2};

    Result captureHop(U64 hop, std::span<CF32> samples, U64& segments, SweepCapture& capture,
                      const SweepRunning& running);
    virtual Result tuneHop(U64 hop);

    Result restore(const Settings& previous, Result failure);
    Result restart(const std::function<Result()>& change);
    U64 guardSamples(const Settings& settings, F64 sampleRate) const;
    U64 nextGeneration();
    Result fail(U64& failures);

    Info deviceInfo;
    Capabilities deviceCapabilities;
    Applied appliedSettings;
    Settings requestedSettings;
    SweepRequest sweepRequest;
    bool sweepPrepared = false;
    F64 defaultSettlingTime = 0.0;
    U64 guardLength = 0;

 private:
    mutable std::mutex statisticsMutex;
    Statistics deviceStatistics;
    std::atomic<U64> currentGeneration{0};
    std::vector<CF32> sweepBuffer;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_DEVICE_HH
