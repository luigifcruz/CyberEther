#include "base.hh"
#include "tools/library.hh"

#include "../sweep.hh"
#include "../../tools/timestamp_filter.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

#include <LimeSuite.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

using Status = Device::Status;
using Clock = std::chrono::steady_clock;

constexpr F32 Latency = 0.15f;
constexpr F64 SettlingTime = 2.0e-3;
constexpr unsigned MaxGain = 73;
constexpr U64 PacketSamples = 1360;
constexpr U64 DrainSamples = 65536;
constexpr U64 UsbTransfers = 16;
constexpr U32 FifoSamples = 1 << 20;
constexpr F64 SweepSampleRate = 20.0e6;
constexpr F64 MinCalibrationBandwidth = 2.5e6;

std::chrono::milliseconds Remaining(const Clock::time_point deadline) {
    const auto left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    return std::max(left, std::chrono::milliseconds(1));
}

class LimeSdrSweep final : public Sweep::Retune {
 public:
    ~LimeSdrSweep() override;

    Result connect(const Device::Info& info);
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(U64 hop) override;
    Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;

 private:
    void disconnect();
    Result setOscillator(F64 frequency, bool useCache);
    Result tuneOscillator(F64 frequency, bool useCache);
    bool poll();
    Result applyGain(F64 gain, bool force);

    lms_device_t* handle = nullptr;
    std::vector<int> antennaIndices;
    lms_stream_t stream = {};
    U64 streamBatch = 1;
    bool streamReady = false;
    bool streaming = false;
    U64 queueGeneration = 0;
    Clock::time_point received{};
    TimestampFilter timestamps;
    std::vector<CF32> drain;
    bool cachedTuning = false;
    bool calibrated = false;
};

LimeSdrSweep::~LimeSdrSweep() {
    disconnect();
}

void LimeSdrSweep::disconnect() {
    close();
    if (handle) {
        if (streamReady) {
            LMS_DestroyStream(handle, &stream);
            streamReady = false;
        }
        LMS_Close(handle);
        handle = nullptr;
        --LimeSdrLibrary::OpenDevices();
    }
    calibrated = false;
}

Result LimeSdrSweep::connect(const Device::Info& info) {
    std::vector<std::string> matches;
    for (const auto& board : LimeSdrLibrary::Boards()) {
        if (board.serial == info.serial) {
            matches.push_back(board.serialize());
        }
    }
    if (matches.empty()) {
        JST_ERROR("[SDR_LIMESDR] Device with serial '{}' is no longer available.", info.serial);
        return Result::INCOMPLETE;
    }
    if (matches.size() != 1) {
        JST_ERROR("[SDR_LIMESDR] Several boards report serial '{}'.", info.serial);
        return Result::INCOMPLETE;
    }
    const auto& match = matches.front();
    lms_info_str_t selected = {};
    std::strncpy(selected, match.c_str(), sizeof(selected) - 1);
    if (LMS_Open(&handle, selected, nullptr) != 0) {
        handle = nullptr;
        JST_ERROR("[SDR_LIMESDR] Failed to open device '{}': {}", info.label,
                  LMS_GetLastErrorMessage());
        return Result::ERROR;
    }
    ++LimeSdrLibrary::OpenDevices();
    deviceInfo = info;

    if (LMS_Init(handle) != 0 || LMS_EnableChannel(handle, LMS_CH_RX, 0, true) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to initialize the device: {}", LMS_GetLastErrorMessage());
        return Result::ERROR;
    }

    lms_range_t frequency = {};
    lms_range_t rate = {};
    if (LMS_GetLOFrequencyRange(handle, LMS_CH_RX, &frequency) != 0 ||
        LMS_GetSampleRateRange(handle, LMS_CH_RX, &rate) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to query device ranges.");
        return Result::ERROR;
    }

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = {{frequency.min, frequency.max}};
    caps.sampleRateRanges = {{rate.min, rate.max}};
    caps.antennas.clear();
    antennaIndices.clear();
    const int antennaCount = LMS_GetAntennaList(handle, LMS_CH_RX, 0, nullptr);
    if (antennaCount > 0) {
        auto names = std::make_unique<lms_name_t[]>(static_cast<std::size_t>(antennaCount));
        LMS_GetAntennaList(handle, LMS_CH_RX, 0, names.get());
        for (int i = 0; i < antennaCount; ++i) {
            if (std::string(names[i]) == "NONE") {
                continue;
            }
            caps.antennas.emplace_back(names[i]);
            antennaIndices.push_back(i);
        }
    }
    caps.automaticGain = false;
    caps.biasTee = false;
    caps.sweepSampleRate = std::min(rate.max, SweepSampleRate);
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    defaultSettlingTime = SettlingTime;
    appliedSettings.gainRange = Device::Range{0.0, static_cast<F64>(MaxGain)};

    JST_DEBUG("[SDR_LIMESDR] Opened '{}' at {}.", info.label, match);
    return Result::SUCCESS;
}

Result LimeSdrSweep::apply(const Settings& settings) {
    if (streamReady) {
        disconnect();
    }
    if (!handle) {
        const auto info = deviceInfo;
        if (const auto result = connect(info); result != Result::SUCCESS) {
            disconnect();
            return result;
        }
    }
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_LIMESDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    const auto& antennas = deviceCapabilities.antennas;
    const auto antenna = std::find(antennas.begin(), antennas.end(), settings.antenna);
    if (!settings.antenna.empty() && antenna == antennas.end()) {
        JST_ERROR("[SDR_LIMESDR] Receive antenna '{}' is not supported.", settings.antenna);
        return Result::ERROR;
    }
    if (settings.automaticGain) {
        JST_WARN("[SDR_LIMESDR] LimeSDR has no automatic gain control. Using the manual gain.");
    }
    if (settings.biasTee) {
        JST_WARN("[SDR_LIMESDR] Bias-T is not supported by LimeSDR.");
    }

    {
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        if (LMS_SetSampleRate(handle, settings.sampleRate, 0) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set the sample rate: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        if (antenna != antennas.end() &&
            LMS_SetAntenna(handle, LMS_CH_RX, 0,
                           antennaIndices[antenna - antennas.begin()]) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to select the antenna: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        JST_CHECK(applyGain(settings.gain, true));
        lms_range_t filter = {};
        if (LMS_GetLPFBWRange(handle, LMS_CH_RX, &filter) != 0 ||
            LMS_SetLPFBW(handle, LMS_CH_RX, 0,
                         std::clamp(settings.sampleRate, filter.min, filter.max)) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set the receive filter: {}",
                      LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
    }
    calibrated = false;

    F64 hostRate = 0.0;
    F64 rfRate = 0.0;
    if (LMS_GetSampleRate(handle, LMS_CH_RX, 0, &hostRate, &rfRate) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read back the sample rate: {}",
                  LMS_GetLastErrorMessage());
        return Result::ERROR;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = hostRate;
    appliedSettings.antenna = settings.antenna;
    appliedSettings.automaticGain = false;
    appliedSettings.biasTee = false;
    return Result::SUCCESS;
}

Result LimeSdrSweep::setOscillator(const F64 frequency, const bool useCache) {
    bool tuned = false;
    {
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        if (useCache) {
            LMS_EnableCache(handle, true);
        }
        tuned = LMS_SetLOFrequency(handle, LMS_CH_RX, 0, frequency) == 0;
        if (useCache) {
            LMS_EnableCache(handle, false);
        }
    }
    if (!tuned) {
        JST_ERROR("[SDR_LIMESDR] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                  LMS_GetLastErrorMessage());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result LimeSdrSweep::tuneOscillator(const F64 frequency, const bool useCache) {
    JST_CHECK(setOscillator(frequency, useCache));
    F64 actual = 0.0;
    if (LMS_GetLOFrequency(handle, LMS_CH_RX, 0, &actual) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read back the tuned frequency: {}",
                  LMS_GetLastErrorMessage());
        return Result::ERROR;
    }
    tunedFrequency = actual;
    return Result::SUCCESS;
}

Result LimeSdrSweep::open() {
    if (streamReady || !handle) {
        const auto settings = requestedSettings;
        JST_CHECK(apply(settings));
    }
    cachedTuning = LimeSdrLibrary::OpenDevices() == 1;
    if (!cachedTuning) {
        JST_WARN("[SDR_LIMESDR] Several LimeSDR devices are open. Cached tuning maps are "
                 "shared across the process, so full tuning is used.");
    }
    JST_CHECK(tuneOscillator(sweepRequest.frequencies.front(), false));
    if (!calibrated) {
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        const F64 bandwidth = std::max(requestedSettings.sampleRate, MinCalibrationBandwidth);
        calibrated = LMS_Calibrate(handle, LMS_CH_RX, 0, bandwidth, 0) == 0;
        if (!calibrated) {
            JST_WARN("[SDR_LIMESDR] Failed to calibrate the receiver: {}",
                     LMS_GetLastErrorMessage());
        }
    }
    const F64 sampleRate = appliedSettings.sampleRate;
    try {
        drain.resize(DrainSamples);
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_LIMESDR] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    if (!streamReady) {
        stream = {};
        stream.isTx = false;
        stream.channel = 0;
        stream.fifoSize = FifoSamples;
        stream.throughputVsLatency = Latency;
        stream.dataFmt = lms_stream_t::LMS_FMT_F32;
        if (LMS_SetupStream(handle, &stream) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set up the stream: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        const F64 limit = (sampleRate / 1.0e6 + 5.0) * Latency;
        streamBatch = 1;
        for (U64 batch = 1; static_cast<F64>(batch) < limit; batch *= 2) {
            streamBatch = batch;
        }
        streamReady = true;
    }
    if (LMS_StartStream(&stream) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to start the stream: {}", LMS_GetLastErrorMessage());
        return Result::ERROR;
    }
    lms_stream_status_t status = {};
    if (LMS_GetStreamStatus(&stream, &status) != 0) {
        LMS_StopStream(&stream);
        JST_ERROR("[SDR_LIMESDR] Failed to read the stream status.");
        return Result::ERROR;
    }
    queueGeneration = nextGeneration();
    received = Clock::now();
    timestamps.restart(guard(), static_cast<U64>(sampleRate));
    streaming = true;
    return Result::SUCCESS;
}

void LimeSdrSweep::close() {
    if (streaming) {
        LMS_StopStream(&stream);
    }
    streaming = false;
}

Result LimeSdrSweep::tune(const U64 hop) {
    if (!streaming) {
        JST_ERROR("[SDR_LIMESDR] Cannot retune a stopped sweep.");
        return Result::ERROR;
    }
    const bool useCache = cachedTuning && LimeSdrLibrary::OpenDevices() == 1;
    JST_CHECK(tuneOscillator(sweepRequest.frequencies[hop], useCache));
    queueGeneration = nextGeneration();

    lms_stream_status_t status = {};
    if (LMS_GetStreamStatus(&stream, &status) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read the stream position after retuning.");
        return Result::ERROR;
    }
    const U64 settle = guard();
    const U64 margin = PacketSamples * (streamBatch * UsbTransfers + 2);
    if (settle > std::numeric_limits<U64>::max() - margin ||
        status.timestamp > std::numeric_limits<U64>::max() - margin - settle) {
        JST_ERROR("[SDR_LIMESDR] Settling guard exceeds the hardware timestamp range.");
        return Result::ERROR;
    }
    timestamps.retune(status.timestamp + margin + settle);
    return Result::SUCCESS;
}

bool LimeSdrSweep::poll() {
    lms_stream_status_t status = {};
    if (LMS_GetStreamStatus(&stream, &status) != 0) {
        return false;
    }
    if (status.droppedPackets == 0 && status.overrun == 0) {
        return true;
    }
    const U64 margin = 2 * PacketSamples;
    timestamps.lose(status.timestamp > std::numeric_limits<U64>::max() - margin
        ? std::numeric_limits<U64>::max()
        : status.timestamp + margin);
    return true;
}

Device::Capture LimeSdrSweep::read(std::span<CF32> samples,
                                   const std::chrono::milliseconds timeout) {
    const auto finish = [&](Device::Capture capture) {
        capture.generation = queueGeneration;
        capture.frequency = tunedFrequency;
        capture.sampleRate = appliedSettings.sampleRate;
        capture.epoch = timestamps.epoch();
        return capture;
    };
    const auto receive = [&](CF32* data, const U64 count, lms_stream_meta_t& meta,
                             const std::chrono::milliseconds wait) {
        const int delivered = LMS_RecvStream(&stream, data, count, &meta,
                                             static_cast<unsigned>(wait.count()));
        if (delivered > 0) {
            received = Clock::now();
        }
        return delivered;
    };
    const auto blind = [&] {
        return finish({.status = Status::Failure, .error = "failed to read the stream status"});
    };
    const auto idle = [&] {
        if (Clock::now() - received > Device::StallTimeout) {
            return finish({.status = Status::Disconnected,
                           .error = "LimeSDR stopped delivering samples"});
        }
        return finish({.status = Status::Timeout});
    };

    if (!streaming) {
        return {.status = Status::Failure, .error = "stream is not running"};
    }

    const auto deadline = Clock::now() + timeout;
    while (const U64 pending = timestamps.pending()) {
        const U64 count = std::min<U64>(pending, drain.size());
        lms_stream_meta_t meta = {};
        const int delivered = receive(drain.data(), count, meta, Remaining(deadline));
        if (delivered < 0) {
            return finish({.status = Status::Disconnected, .error = LMS_GetLastErrorMessage()});
        }
        if (!poll()) {
            return blind();
        }
        if (delivered == 0) {
            return idle();
        }
        const auto decision = timestamps.accept(meta.timestamp, static_cast<U64>(delivered));
        if (decision.status == Status::Failure || decision.status == Status::Discontinuity) {
            return finish({.status = decision.status, .lost = decision.lost,
                           .error = decision.error});
        }
        if (Clock::now() >= deadline) {
            return finish({.status = Status::Timeout});
        }
    }

    lms_stream_meta_t meta = {};
    const int delivered = receive(samples.data(), samples.size(), meta, Remaining(deadline));
    if (delivered < 0) {
        return finish({.status = Status::Disconnected, .error = LMS_GetLastErrorMessage()});
    }
    if (!poll()) {
        return blind();
    }
    if (delivered == 0) {
        return idle();
    }

    const U64 count = static_cast<U64>(delivered);
    const auto decision = timestamps.accept(meta.timestamp, count);
    Device::Capture capture{.status = decision.status, .lost = decision.lost,
                            .error = decision.error};
    if (decision.status != Status::Samples) {
        return finish(capture);
    }
    if (decision.skip > 0) {
        std::copy(samples.begin() + decision.skip, samples.begin() + count, samples.begin());
    }
    capture.count = count - decision.skip;
    capture.timestamp = meta.timestamp + decision.skip;
    return finish(capture);
}

Result LimeSdrSweep::setAutomaticGain(const bool automaticGain) {
    if (automaticGain) {
        JST_WARN("[SDR_LIMESDR] LimeSDR has no automatic gain control. Using the manual gain.");
    }
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result LimeSdrSweep::setGain(const F64 gain) {
    {
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        JST_CHECK(applyGain(gain, false));
    }
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 LimeSdrSweep::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Sweep::resolveGain(gain, range));
}

Result LimeSdrSweep::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (!force && target == appliedSettings.gain) {
        return Result::SUCCESS;
    }
    if (LMS_SetGaindB(handle, LMS_CH_RX, 0, static_cast<unsigned>(target)) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to set the gain to {:.0f} dB: {}", target,
                  LMS_GetLastErrorMessage());
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result LimeSdr::Sweep(const Device::Info& info, std::unique_ptr<Radio::Sweep>& sweep) {
    auto created = std::make_unique<LimeSdrSweep>();
    JST_CHECK(created->connect(info));
    sweep = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
