#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_PLUTOSDR_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_PLUTOSDR_HH

#include <memory>
#include <vector>

#include "../device.hh"

namespace Jetstream::Radio {

struct PlutoSdrDevice : public Device {
 public:
    PlutoSdrDevice();
    ~PlutoSdrDevice() override;

    static std::vector<Info> Enumerate();
    static Result Create(const Info& info, std::unique_ptr<Device>& device);

    Result start(const Settings& settings) override;
    void stop() override;
    Result tune(F64 frequency) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;
    F64 resolveGain(F64 gain, const std::optional<Range>& range) const override;
    Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;
    Result runSweep(const SweepRunning& running, const SweepSink& sink) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_PLUTOSDR_HH
