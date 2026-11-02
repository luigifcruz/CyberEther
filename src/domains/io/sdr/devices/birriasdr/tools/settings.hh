#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_BIRRIASDR_TOOLS_SETTINGS_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_BIRRIASDR_TOOLS_SETTINGS_HH

#include <algorithm>
#include <array>
#include <utility>

#include <birriasdr.h>

#include "../../device.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Radio::BirriaSdrSettings {

inline constexpr std::array<U32, 4> SampleRates = {10000000, 20000000, 40000000, 80000000};
inline constexpr std::array<U8, 3> MaxStageGains = {5, 8, 12};
inline constexpr U64 MaxGain = MaxStageGains[0] + MaxStageGains[1] + MaxStageGains[2];

#if defined(__APPLE__)
inline constexpr U32 InFlightTransfers = 256;
#else
inline constexpr U32 InFlightTransfers = 64;
#endif

inline constexpr U32 TransferSamples = 16384;
inline constexpr U64 SramPackets = 32;
inline constexpr U64 PacketsPerTransfer = 8;
inline constexpr U64 HostQueueTransfers = InFlightTransfers < 64 ? InFlightTransfers * 8 : 512;

inline U64 StaleSamples(const F64 sampleRate) {
    const U64 packetSamples = sampleRate <= 20.0e6 ? 1024 : 2048;
    const U64 transfers = InFlightTransfers + HostQueueTransfers;
    const U64 packets = SramPackets + transfers * PacketsPerTransfer + 1;
    return packets * packetSamples + TransferSamples;
}

inline birriasdr_gain StageGains(U64 total) {
    std::array<U8, 3> stages = {0, 0, 0};
    for (U64 i = 0; i < stages.size(); ++i) {
        stages[i] = static_cast<U8>(std::min<U64>(total, MaxStageGains[i]));
        total -= stages[i];
    }
    return {.lna = stages[0], .lpf = stages[1], .digital = stages[2]};
}

template<typename GetBand>
Result ReadCapabilities(const GetBand& getBand, const F64 defaultRate, Device::Capabilities& caps) {
    Device::Capabilities discovered;
    for (U32 index = 0; ; ++index) {
        uint64_t minimum = 0;
        uint64_t maximum = 0;
        const int status = getBand(index, minimum, maximum);
        if (status == BIRRIASDR_ERROR_INVALID) {
            break;
        }
        if (status == BIRRIASDR_ERROR_UNSUPPORTED) {
            continue;
        }
        if (status != BIRRIASDR_OK || minimum == 0 || maximum < minimum) {
            JST_ERROR("[SDR_BIRRIASDR] Failed to query tuning band {}: {}", index,
                      birriasdr_strerror(status == BIRRIASDR_OK ? BIRRIASDR_ERROR_STATE : status));
            return Result::ERROR;
        }
        discovered.frequencyRanges.push_back(
            {static_cast<F64>(minimum), static_cast<F64>(maximum)});
    }
    if (discovered.frequencyRanges.empty()) {
        JST_ERROR("[SDR_BIRRIASDR] No supported tuning bands.");
        return Result::ERROR;
    }
    if (std::find(SampleRates.begin(), SampleRates.end(), defaultRate) == SampleRates.end()) {
        JST_ERROR("[SDR_BIRRIASDR] Invalid default sample rate ({:.3f} MHz).", defaultRate / 1e6);
        return Result::ERROR;
    }
    for (const auto rate : SampleRates) {
        discovered.sampleRateRanges.push_back({static_cast<F64>(rate), static_cast<F64>(rate)});
    }
    discovered.automaticGain = true;
    discovered.biasTee = false;
    discovered.gainUnit = "steps";
    discovered.sweepSampleRate = defaultRate;
    discovered.usableBandwidth = 0.8;
    discovered.dcExclusion = 20.0e3;
    caps = std::move(discovered);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio::BirriaSdrSettings

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_BIRRIASDR_TOOLS_SETTINGS_HH
