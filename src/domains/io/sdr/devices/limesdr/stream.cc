#include "base.hh"
#include "tools/library.hh"

#include "../stream.hh"
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

constexpr F32 Latency = 0.5f;
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

class LimeSdrStream final : public Stream {
 public:
    explicit LimeSdrStream(LmsApi& api) : api(&api) {}
    ~LimeSdrStream() override;

    Result connect(const Device::Info& info);
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;
    Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(F64 frequency) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;

 private:
    void disconnect();
    Result setOscillator(F64 frequency, bool useCache);
    Result tuneOscillator(F64 frequency, bool useCache);
    bool poll();
    Result applyGain(F64 gain, bool force);

    LmsApi* api;
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
};

LimeSdrStream::~LimeSdrStream() {
    disconnect();
}

void LimeSdrStream::disconnect() {
    close();
    if (handle) {
        if (streamReady) {
            api->DestroyStream(handle, &stream);
            streamReady = false;
        }
        api->Close(handle);
        handle = nullptr;
        --api->openDevices;
    }
}

Result LimeSdrStream::connect(const Device::Info& info) {
    std::vector<std::string> matches;
    for (const auto& board : api->boards()) {
        if (board.serial == info.serial) {
            matches.push_back(board.address);
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
    if (api->Open(&handle, selected, nullptr) != 0) {
        handle = nullptr;
        JST_ERROR("[SDR_LIMESDR] Failed to open device '{}': {}", info.label,
                  api->GetLastErrorMessage());
        return Result::ERROR;
    }
    ++api->openDevices;
    deviceInfo = info;

    if (api->Init(handle) != 0 || api->EnableChannel(handle, LMS_CH_RX, 0, true) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to initialize the device: {}", api->GetLastErrorMessage());
        return Result::ERROR;
    }

    lms_range_t frequency = {};
    lms_range_t rate = {};
    if (api->GetLOFrequencyRange(handle, LMS_CH_RX, &frequency) != 0 ||
        api->GetSampleRateRange(handle, LMS_CH_RX, &rate) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to query device ranges.");
        return Result::ERROR;
    }

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = {{frequency.min, frequency.max}};
    caps.sampleRateRanges = {{rate.min, rate.max}};
    caps.antennas.clear();
    antennaIndices.clear();
    for (auto& [index, name] : api->antennas(handle)) {
        if (name == "NONE") {
            continue;
        }
        caps.antennas.push_back(std::move(name));
        antennaIndices.push_back(index);
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

Result LimeSdrStream::apply(const Settings& settings) {
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
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_LIMESDR] Frequency ({:.3f} MHz) not supported. Use {}.",
                  settings.frequency / 1e6,
                  Device::Range::Describe(deviceCapabilities.frequencyRanges));
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
        std::lock_guard lock(api->tuningMutex);
        if (api->SetSampleRate(handle, settings.sampleRate, 0) != 0 ||
            api->SetLOFrequency(handle, LMS_CH_RX, 0, settings.frequency) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to configure device: {}", api->GetLastErrorMessage());
            return Result::ERROR;
        }
        if (antenna != antennas.end() &&
            api->SetAntenna(handle, LMS_CH_RX, 0,
                           antennaIndices[antenna - antennas.begin()]) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to select the antenna: {}", api->GetLastErrorMessage());
            return Result::ERROR;
        }
        JST_CHECK(applyGain(settings.gain, true));
        lms_range_t filter = {};
        if (api->GetLPFBWRange(handle, LMS_CH_RX, &filter) != 0 ||
            api->SetLPFBW(handle, LMS_CH_RX, 0,
                         std::clamp(settings.sampleRate, filter.min, filter.max)) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set the receive filter: {}",
                      api->GetLastErrorMessage());
            return Result::ERROR;
        }
        const F64 bandwidth = std::max(settings.sampleRate, MinCalibrationBandwidth);
        if (api->Calibrate(handle, LMS_CH_RX, 0, bandwidth, 0) != 0) {
            JST_WARN("[SDR_LIMESDR] Failed to calibrate the receiver: {}",
                     api->GetLastErrorMessage());
        }
    }

    F64 hostRate = 0.0;
    F64 rfRate = 0.0;
    F64 frequency = 0.0;
    if (api->GetSampleRate(handle, LMS_CH_RX, 0, &hostRate, &rfRate) != 0 ||
        api->GetLOFrequency(handle, LMS_CH_RX, 0, &frequency) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read back the applied settings.");
        return Result::ERROR;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = hostRate;
    appliedSettings.frequency = frequency;
    appliedSettings.antenna = settings.antenna;
    appliedSettings.automaticGain = false;
    appliedSettings.biasTee = false;
    return Result::SUCCESS;
}

Result LimeSdrStream::setOscillator(const F64 frequency, const bool useCache) {
    bool tuned = false;
    {
        std::lock_guard lock(api->tuningMutex);
        if (useCache) {
            api->EnableCache(handle, true);
        }
        tuned = api->SetLOFrequency(handle, LMS_CH_RX, 0, frequency) == 0;
        if (useCache) {
            api->EnableCache(handle, false);
        }
    }
    if (!tuned) {
        JST_ERROR("[SDR_LIMESDR] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                  api->GetLastErrorMessage());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result LimeSdrStream::tuneOscillator(const F64 frequency, const bool useCache) {
    JST_CHECK(setOscillator(frequency, useCache));
    F64 actual = 0.0;
    if (api->GetLOFrequency(handle, LMS_CH_RX, 0, &actual) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read back the tuned frequency: {}",
                  api->GetLastErrorMessage());
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = actual;
    return Result::SUCCESS;
}

Result LimeSdrStream::open() {
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
        if (api->SetupStream(handle, &stream) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set up the stream: {}", api->GetLastErrorMessage());
            return Result::ERROR;
        }
        const F64 limit = (sampleRate / 1.0e6 + 5.0) * Latency;
        for (U64 batch = 1; static_cast<F64>(batch) < limit; batch *= 2) {
            streamBatch = batch;
        }
        streamReady = true;
    }
    if (api->StartStream(&stream) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to start the stream: {}", api->GetLastErrorMessage());
        return Result::ERROR;
    }
    lms_stream_status_t status = {};
    if (api->GetStreamStatus(&stream, &status) != 0) {
        api->StopStream(&stream);
        JST_ERROR("[SDR_LIMESDR] Failed to read the stream status.");
        return Result::ERROR;
    }
    queueGeneration = nextGeneration();
    received = Clock::now();
    timestamps.restart(guard(), static_cast<U64>(sampleRate));
    streaming = true;
    return Result::SUCCESS;
}

void LimeSdrStream::close() {
    if (streaming) {
        api->StopStream(&stream);
    }
    streaming = false;
}

Result LimeSdrStream::tune(const F64 frequency) {
    JST_CHECK(tuneOscillator(frequency, false));
    if (!streaming) {
        return Result::SUCCESS;
    }
    queueGeneration = nextGeneration();

    lms_stream_status_t status = {};
    if (api->GetStreamStatus(&stream, &status) != 0) {
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

bool LimeSdrStream::poll() {
    lms_stream_status_t status = {};
    if (api->GetStreamStatus(&stream, &status) != 0) {
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

Device::Capture LimeSdrStream::read(std::span<CF32> samples,
                                    const std::chrono::milliseconds timeout) {
    const auto finish = [&](Device::Capture capture) {
        capture.generation = queueGeneration;
        capture.frequency = appliedSettings.frequency;
        capture.sampleRate = appliedSettings.sampleRate;
        capture.epoch = timestamps.epoch();
        return capture;
    };
    const auto receive = [&](CF32* data, const U64 count, lms_stream_meta_t& meta,
                             const std::chrono::milliseconds wait) {
        const int delivered = api->RecvStream(&stream, data, count, &meta,
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
            return finish({.status = Status::Disconnected, .error = api->GetLastErrorMessage()});
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
        return finish({.status = Status::Disconnected, .error = api->GetLastErrorMessage()});
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

Result LimeSdrStream::setAutomaticGain(const bool automaticGain) {
    if (automaticGain) {
        JST_WARN("[SDR_LIMESDR] LimeSDR has no automatic gain control. Using the manual gain.");
    }
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result LimeSdrStream::setGain(const F64 gain) {
    {
        std::lock_guard lock(api->tuningMutex);
        JST_CHECK(applyGain(gain, false));
    }
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 LimeSdrStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range));
}

Result LimeSdrStream::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (!force && target == appliedSettings.gain) {
        return Result::SUCCESS;
    }
    if (api->SetGaindB(handle, LMS_CH_RX, 0, static_cast<unsigned>(target)) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to set the gain to {:.0f} dB: {}", target,
                  api->GetLastErrorMessage());
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

Result OpenStream(LmsApi& api, const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<LimeSdrStream>(api);
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace

Result LimeSdr::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    return OpenStream(LmsApi::Bundled(), info, stream);
}

Result LimeSuiteNg::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto* api = LmsApi::Ng();
    if (!api) {
        JST_ERROR("[SDR_LIMESDR] LimeSuiteNG is not installed.");
        return Result::ERROR;
    }
    return OpenStream(*api, info, stream);
}

}  // namespace Jetstream::Radio
