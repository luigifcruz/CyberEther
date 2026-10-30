#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_SWEEP_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_SWEEP_HH

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "device.hh"

namespace Jetstream::Radio {

class JETSTREAM_API Sweep {
 public:
    struct Settings {
        F64 sampleRate = 2.0e6;
        std::string antenna;
        bool automaticGain = true;
        bool biasTee = false;
        F64 settlingTime = -1.0;
        F64 gain = 30.0;
    };

    struct Applied {
        F64 sampleRate = 0.0;
        std::string antenna;
        bool automaticGain = false;
        bool biasTee = false;
        F64 gain = 0.0;
        std::optional<Device::Range> gainRange;
    };

    struct Request {
        bool firmware = false;
        std::vector<F64> frequencies;
        F64 sampleRate = 0.0;
        U64 captureLength = 0;
        U64 dwell = 1;
        F64 startFrequency = 0.0;
        F64 endFrequency = 0.0;

        bool operator==(const Request&) const = default;
    };

    struct Capture {
        U64 hop = 0;
        F64 frequency = 0.0;
        std::span<const CF32> samples;
    };

    struct Statistics {
        U64 failedHops = 0;
    };

    using Sink = std::function<bool(const Capture&)>;
    using Running = std::function<bool()>;

    class Retune;

    virtual ~Sweep() = default;

    static Result Open(const std::string& selector, std::unique_ptr<Sweep>& sweep);

    const Device::Info& info() const;
    const Device::Capabilities& capabilities() const;
    const Settings& settings() const;
    const Applied& applied() const;
    const Request& request() const;
    Statistics statistics() const;
    virtual F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const;

    Result configure(const Settings& settings);
    Result start(const Request& request);
    Result update(const Settings& settings);
    virtual Result run(const Running& running, const Sink& sink) = 0;

 protected:
    virtual Result apply(const Settings& settings) = 0;
    virtual Result open() = 0;
    virtual void close() = 0;
    virtual Result setAutomaticGain(bool automaticGain) = 0;
    virtual Result setGain(F64 gain) = 0;

    U64 guard() const;
    Result fail(U64& failures);

    Device::Info deviceInfo;
    Device::Capabilities deviceCapabilities;
    Settings requestedSettings;
    Applied appliedSettings;
    Request sweepRequest;
    F64 defaultSettlingTime = 0.0;

 private:
    Result setSettlingTime(F64 settlingTime);

    std::atomic<U64> failedHops{0};
};

class JETSTREAM_API Sweep::Retune : public Sweep {
 public:
    Result run(const Running& running, const Sink& sink) override;

 protected:
    virtual Result tune(U64 hop) = 0;
    virtual Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) = 0;

    Result capture(U64 hop, std::span<CF32> samples, U64& segments, Capture& capture,
                   const Running& running);
    U64 generation() const;
    U64 nextGeneration();

    F64 tunedFrequency = 0.0;

 private:
    std::vector<CF32> buffer;
    std::atomic<U64> currentGeneration{0};
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_SWEEP_HH
