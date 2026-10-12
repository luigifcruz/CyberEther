#ifndef JETSTREAM_DOMAINS_IO_SDR_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_SDR_MODULE_IMPL_HH

#include <atomic>
#include <memory>
#include <thread>

#include <jetstream/domains/io/sdr/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

#include "domains/io/sdr/devices/device.hh"
#include "domains/io/sdr/tools/control_channel.hh"
#include "domains/io/sdr/tools/stream_buffer.hh"

namespace Jetstream::Modules {

struct JETSTREAM_API SdrImpl : public Module::Impl, public DynamicConfig<Sdr> {
 public:
    struct LossSnapshot {
        F64 bufferLoss = 0.0;
        U64 deviceGaps = 0;
        U64 deviceLostSamples = 0;
    };

    ~SdrImpl() override;

    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    F32 getBufferHealth() const;
    LossSnapshot getLoss() const;
    std::pair<F32, F32> getThroughput() const;
    std::vector<std::string> listAntennas() const;
    Radio::Device::Applied getApplied() const;
    std::optional<Radio::Device::Range> getGainRange() const;

 protected:
    Tensor buffer;

    U64 validatedOutputSizeBytes = 0;
    U64 validatedInternalElements = 0;
    U64 validatedInternalSizeBytes = 0;

    std::thread receiver;
    std::atomic<bool> errored{false};
    std::atomic<bool> streaming{false};

    Radio::StreamBuffer stream;
    Radio::ControlChannel control;
    Radio::StreamBuffer::Metadata publishedMetadata;

    Tools::Snapshot<F32> bufferHealth{0.0f};
    Tools::Snapshot<std::pair<F32, F32>> throughput{{0.0f, 0.0f}};
    Tools::Snapshot<U64> deviceGaps{0};
    Tools::Snapshot<U64> deviceLostSamples{0};
    Tools::Snapshot<Radio::Device::Applied> published;

    Result receiveLoop();
    Result emitFrame();
    Result publishMetadata(const Radio::StreamBuffer::Metadata& metadata);

 private:
    std::unique_ptr<Radio::Device> radio;
    std::vector<std::string> antennas;
    Radio::Device::Settings tuned;

    Radio::Device::Settings settingsOf(const Sdr& config) const;
    Result applySettings(const Radio::Device::Settings& desired);
    void resetStream();
    Result startReceiver();
    void stopReceiver();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SDR_MODULE_IMPL_HH
