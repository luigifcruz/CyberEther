#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_DEVICE_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_DEVICE_HH

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

class Stream;
class Sweep;

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

    struct Layout {
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

    struct Firmware {
        using Planner = std::function<bool(F64 startFrequency, F64 endFrequency,
                                           U64 captureLength, Layout& layout)>;

        F64 sampleRate = 0.0;
        U64 maxDwell = 0;
        U64 maxCaptureLength = 0;
        U64 captureBase = 1;
        U64 captureStep = 1;
        Planner layout;
    };

    struct JETSTREAM_API Capabilities {
        std::vector<Range> frequencyRanges;
        std::vector<Range> sampleRateRanges;
        std::vector<std::string> antennas;
        bool automaticGain = false;
        bool biasTee = false;
        std::string gainUnit = "dB";
        F64 sweepSampleRate = 0.0;
        F64 usableBandwidth = 0.75;
        F64 dcExclusion = 0.0;
        std::optional<Firmware> firmwareSweep;

        std::vector<F64> fixedSampleRates() const;
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

    struct Driver {
        std::string name;
        std::function<std::vector<Info>()> enumerate;
        std::function<Result(const Info& info, std::unique_ptr<Stream>& stream)> stream;
        std::function<Result(const Info& info, std::unique_ptr<Sweep>& sweep)> sweep;
    };

    using Labels = std::vector<std::pair<std::string, Info>>;

    static constexpr std::chrono::seconds StallTimeout{2};

    static std::vector<Driver> Drivers();
    static void Override(std::optional<std::vector<Driver>> drivers);
    static std::vector<Info> List(bool refresh = false);
    static void ClearCache();
    static Labels Label(const std::vector<Info>& devices);
    static std::optional<Capabilities> Cached(const std::string& selector);
    static bool SupportsBiasTee(const std::string& selector);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_DEVICE_HH
