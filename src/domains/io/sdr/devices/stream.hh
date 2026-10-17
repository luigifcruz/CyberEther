#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_STREAM_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_STREAM_HH

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "device.hh"

namespace Jetstream::Radio {

class JETSTREAM_API Stream {
 public:
    struct Settings {
        F64 frequency = 96.9e6;
        F64 sampleRate = 2.0e6;
        std::string antenna;
        bool automaticGain = true;
        bool biasTee = false;
        F64 settlingTime = -1.0;
        F64 gain = 30.0;
    };

    struct Applied {
        F64 frequency = 0.0;
        F64 sampleRate = 0.0;
        std::string antenna;
        bool automaticGain = false;
        bool biasTee = false;
        F64 gain = 0.0;
        std::optional<Device::Range> gainRange;
    };

    virtual ~Stream() = default;

    static Result Open(const std::string& selector, std::unique_ptr<Stream>& stream);

    const Device::Info& info() const;
    const Device::Capabilities& capabilities() const;
    const Settings& settings() const;
    const Applied& applied() const;
    U64 generation() const;
    virtual F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const;

    Result start(const Settings& settings);
    Result update(const Settings& settings);
    virtual Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) = 0;

 protected:
    virtual Result apply(const Settings& settings) = 0;
    virtual Result open() = 0;
    virtual void close() = 0;
    virtual Result tune(F64 frequency) = 0;
    virtual Result setAutomaticGain(bool automaticGain) = 0;
    virtual Result setGain(F64 gain) = 0;

    U64 guard() const;
    U64 nextGeneration();

    Device::Info deviceInfo;
    Device::Capabilities deviceCapabilities;
    Settings requestedSettings;
    Applied appliedSettings;
    F64 defaultSettlingTime = 0.0;

 private:
    Result configure(const Settings& settings);
    Result setSettlingTime(F64 settlingTime);
    Result restart(const Settings& settings, const Settings& previous);
    Result restore(const Settings& previous, Result failure);

    std::atomic<U64> currentGeneration{0};
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_STREAM_HH
