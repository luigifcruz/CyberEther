#ifndef JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH

#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <thread>
#include <vector>

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>

#include <jetstream/domains/io/soapy/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

inline bool SoapyRangeContains(const std::vector<SoapySDR::Range>& ranges, const F32 value) {
    for (const auto& range : ranges) {
        const F32 minimum = static_cast<F32>(range.minimum());
        const F32 maximum = static_cast<F32>(range.maximum());
        if (value < minimum || value > maximum) {
            continue;
        }

        const double step = range.step();
        if (!std::isfinite(step) || step <= 0.0) {
            return true;
        }

        const double stepCount = std::round((static_cast<double>(value) - range.minimum()) / step);
        const F32 closest = static_cast<F32>(range.minimum() + stepCount * step);
        if (value == closest) {
            return true;
        }
    }
    return false;
}

inline Result ValidateSoapyConfig(const F32 frequency,
                                  const F32 sampleRate,
                                  const U64 numberOfBatches,
                                  const U64 numberOfTimeSamples,
                                  const U64 bufferMultiplier,
                                  const std::vector<SoapySDR::Range>* sampleRateRanges = nullptr,
                                  const std::vector<SoapySDR::Range>* frequencyRanges = nullptr) {
    if (!std::isfinite(frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency must be finite.");
        return Result::ERROR;
    }

    if (!std::isfinite(sampleRate) || sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_SOAPY] Sample rate must be finite and positive.");
        return Result::ERROR;
    }

    if (numberOfBatches == 0) {
        JST_ERROR("[MODULE_SOAPY] Number of batches cannot be zero.");
        return Result::ERROR;
    }

    if (numberOfTimeSamples == 0) {
        JST_ERROR("[MODULE_SOAPY] Number of time samples cannot be zero.");
        return Result::ERROR;
    }

    if (bufferMultiplier == 0) {
        JST_ERROR("[MODULE_SOAPY] Buffer multiplier cannot be zero.");
        return Result::ERROR;
    }

    constexpr U64 maxElements = std::numeric_limits<std::size_t>::max() / sizeof(CF32);
    if (numberOfBatches > maxElements / numberOfTimeSamples) {
        JST_ERROR("[MODULE_SOAPY] Output buffer dimensions are too large.");
        return Result::ERROR;
    }

    const U64 outputElements = numberOfBatches * numberOfTimeSamples;
    if (outputElements > maxElements / bufferMultiplier) {
        JST_ERROR("[MODULE_SOAPY] Internal buffer dimensions are too large.");
        return Result::ERROR;
    }

    if (sampleRateRanges != nullptr && !sampleRateRanges->empty() &&
        !SoapyRangeContains(*sampleRateRanges, sampleRate)) {
        JST_ERROR("[MODULE_SOAPY] Sample rate ({:.2f} MHz) not supported.",
                  sampleRate / 1e6);
        return Result::ERROR;
    }

    if (frequencyRanges != nullptr && !frequencyRanges->empty() &&
        !SoapyRangeContains(*frequencyRanges, frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency ({:.2f} MHz) not supported.",
                  frequency / 1e6);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

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
    std::atomic<F32> activeSampleRate{0.0f};

    Tools::CircularBuffer<CF32> circularBuffer;
    Tools::Snapshot<F32> bufferHealth{0.0f};
    Tools::Snapshot<std::pair<F32, F32>> throughput{{0.0f, 0.0f}};

    Result soapyThreadLoop();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_MODULE_IMPL_HH
