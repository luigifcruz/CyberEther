#ifndef JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH

#include <atomic>
#include <map>
#include <thread>
#include <SoapySDR/Types.hpp>

#include <jetstream/domains/io/soapy/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

#include "soapysdr.hh"

namespace Jetstream::Modules {

struct SoapyImpl : public Module::Impl, public DynamicConfig<Soapy> {
 public:
    using DeviceEntry = std::map<std::string, std::string>;
    using DeviceList = std::map<std::string, DeviceEntry>;

    static DeviceList DeviceListFromEntries(const SoapySDR::KwargsList& entries) {
        DeviceList devices;
        for (const auto& entry : entries) {
            const auto labelIt = entry.find("label");
            const auto driverIt = entry.find("driver");
            std::string label = "SoapySDR Device";
            if (labelIt != entry.end() && !labelIt->second.empty()) {
                label = labelIt->second;
            } else if (driverIt != entry.end() && !driverIt->second.empty()) {
                label = driverIt->second;
            }

            std::string uniqueLabel = label;
            if (devices.contains(uniqueLabel)) {
                const auto serialIt = entry.find("serial");
                if (serialIt != entry.end() && !serialIt->second.empty() &&
                    label.find(serialIt->second) == std::string::npos) {
                    uniqueLabel = label + " [" + serialIt->second + "]";
                }

                const std::string uniqueLabelBase = uniqueLabel;
                U64 suffix = 2;
                while (devices.contains(uniqueLabel)) {
                    uniqueLabel = uniqueLabelBase + " #" + std::to_string(suffix++);
                }
            }

            devices.emplace(std::move(uniqueLabel), entry);
        }
        return devices;
    }

    ~SoapyImpl() override;

    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    static Result LoadModulePath(const std::string& path);
    static DeviceList ListAvailableDevices(const std::string& filter = "");
    static std::string DeviceEntryToString(const DeviceEntry& entry);

    F32 getBufferHealth() const;
    F64 getBufferLoss() const;
    std::pair<F32, F32> getThroughput() const;

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
