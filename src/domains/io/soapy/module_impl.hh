#ifndef JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH

#include <thread>
#include <atomic>
#include <vector>
#include <map>

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>

#include <jetstream/domains/io/soapy/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct SoapyImpl : public Module::Impl, public DynamicConfig<Soapy> {
 public:
    using DeviceEntry = std::map<std::string, std::string>;
    using DeviceList = std::map<std::string, DeviceEntry>;

    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    static DeviceList ListAvailableDevices(const std::string& filter = "");
    static std::string DeviceEntryToString(const DeviceEntry& entry);

    F32 getBufferHealth() const;
    std::pair<F32, F32> getThroughput() const;

    Result setTunerFrequency(const F32& frequency);
    Result setSampleRate(const F32& sampleRate);
    Result setAutomaticGain(const bool& automaticGain);

 protected:
    Tensor buffer;

    SoapySDR::Device* soapyDevice = nullptr;
    SoapySDR::Stream* soapyStream = nullptr;

    std::vector<SoapySDR::Range> sampleRateRanges;
    std::vector<SoapySDR::Range> frequencyRanges;

    std::thread producer;
    std::atomic<bool> errored{false};
    std::atomic<bool> streaming{false};

    Tools::CircularBuffer<CF32> circularBuffer;
    Tools::Snapshot<F32> bufferHealth{0.0f};
    Tools::Snapshot<std::pair<F32, F32>> throughput{{0.0f, 0.0f}};

    Result soapyThreadLoop();
    static bool CheckValidRange(const std::vector<SoapySDR::Range>& ranges, const F32& val);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
