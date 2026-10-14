#ifndef JETSTREAM_DOMAINS_IO_SDR_SWEEP_TOOLS_SWEEP_HH
#define JETSTREAM_DOMAINS_IO_SDR_SWEEP_TOOLS_SWEEP_HH

#include <span>
#include <string>
#include <vector>

#include "domains/io/sdr/devices/device.hh"

namespace Jetstream::Modules::Sweep {

constexpr U64 MaxOutputBins = U64{1} << 18;
constexpr U64 MinOutputBins = 16;
constexpr U64 MaxSegmentLength = U64{1} << 16;
constexpr U64 MinSegmentLength = 16;
constexpr U64 MaxHops = U64{1} << 16;
constexpr U64 MaxDwell = 64;
constexpr U64 MaxWorkingSetBytes = U64{256} << 20;

struct JETSTREAM_API Plan {
    enum class Mode : U8 {
        Automatic,
        Retune,
        Firmware,
    };

    struct Run {
        U64 segment = 0;
        U64 grid = 0;
        U64 length = 0;

        bool operator==(const Run&) const = default;
    };

    struct Hop {
        F64 frequency = 0.0;
        std::vector<Run> runs;
    };

    struct Request {
        F64 startFrequency = 0.0;
        F64 endFrequency = 0.0;
        F64 sampleRate = 0.0;
        F64 resolution = 0.0;
        U64 dwell = 1;
        Mode mode = Mode::Automatic;
        F64 settlingTime = 0.0;
        bool flatten = false;
    };

    Mode mode = Mode::Retune;
    F64 sampleRate = 0.0;
    U64 segmentLength = 0;
    U64 dwell = 1;
    F64 binWidth = 0.0;
    F64 startFrequency = 0.0;
    U64 outputBins = 0;
    std::vector<Hop> hops;
    bool flatten = false;

    static Result Create(const Request& request, const Radio::Device::Capabilities& capabilities,
                         Plan& plan);
    static Mode ResolveMode(const Request& request,
                            const Radio::Device::Capabilities& capabilities);
    static std::string ModeName(Mode mode);

    F64 endFrequency() const;
    F64 centerFrequency() const;
    F64 span() const;
    U64 captureSamples() const;
    U64 workingSetBytes() const;
    Radio::Device::SweepRequest deviceRequest() const;
};

class JETSTREAM_API Assembler {
 public:
    Assembler();
    ~Assembler();

    Result configure(const Plan& plan);

    void begin();

    Result add(const Radio::Device::SweepCapture& capture);

    bool complete() const;
    U64 completedHops() const;
    Result synthesize(std::span<CF32> output);

    std::span<const CF32> spectrum() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Modules::Sweep

#endif  // JETSTREAM_DOMAINS_IO_SDR_SWEEP_TOOLS_SWEEP_HH
