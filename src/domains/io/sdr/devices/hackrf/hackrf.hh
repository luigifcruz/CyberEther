#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_HACKRF_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_HACKRF_HH

#include <memory>
#include <vector>

#include "../device.hh"

namespace Jetstream::Radio {

struct HackRfDevice : public Device {
 public:
    HackRfDevice();
    ~HackRfDevice() override;

    static std::vector<Info> Enumerate();
    static Result Create(const Info& info, std::unique_ptr<Device>& device);

    Result start(const Settings& settings) override;
    void stop() override;
    Result tune(F64 frequency) override;
    Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;
    Result prepareSweep(const SweepRequest& request) override;
    Result runSweep(const SweepRunning& running, const SweepSink& sink) override;

 protected:
    Result tuneHop(U64 hop) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_HACKRF_HH
