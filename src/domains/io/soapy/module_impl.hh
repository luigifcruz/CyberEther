#ifndef JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH

#include <atomic>
#include <thread>

#include <jetstream/domains/io/soapy/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

#include "soapysdr.hh"

namespace Jetstream::Modules {

struct JETSTREAM_API SoapyImpl : public Module::Impl, public DynamicConfig<Soapy> {
 public:
    ~SoapyImpl() override;

    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    F32 getBufferHealth() const;
    F64 getBufferLoss() const;
    U64 getDeviceOverflows() const;
    std::pair<F32, F32> getThroughput() const;
    std::vector<std::string> listAntennas() const;

    Result setTunerFrequency(const F32& frequency);
    Result setSampleRate(const F32& sampleRate);
    Result setAutomaticGain(const bool& automaticGain);
    Result setBiasTee(bool enabled);

 protected:
    Tensor buffer;

    U64 validatedOutputSizeBytes = 0;
    U64 validatedInternalElements = 0;
    U64 validatedInternalSizeBytes = 0;

    std::thread producer;
    std::atomic<bool> errored{false};
    std::atomic<bool> streaming{false};
    std::atomic<F32> activeSampleRate{0.0f};

    Tools::CircularBuffer<CF32> circularBuffer;
    Tools::Snapshot<F32> bufferHealth{0.0f};
    Tools::Snapshot<U64> deviceOverflows{0};
    Tools::Snapshot<std::pair<F32, F32>> throughput{{0.0f, 0.0f}};

    Result soapyThreadLoop();

 private:
    SoapyReceiver receiverDevice;

    Result allocateBuffers();
    Result configureDevice(const SoapySDR::Kwargs& streamArgs);
    Result startReceiver();
    void stopReceiver();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
