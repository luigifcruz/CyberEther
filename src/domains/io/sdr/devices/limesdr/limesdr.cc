#include "limesdr.hh"

#include "../../tools/samples.hh"
#include "../../tools/timestamp_filter.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#include <ConnectionRegistry.h>
#include <LimeSuite.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 2.0e-3;
constexpr unsigned MaxGain = 73;
constexpr F32 StreamLatency = 0.5f;
constexpr U64 PacketSamples = 1360;
constexpr U64 DrainSamples = 65536;
constexpr U64 UsbTransfers = 16;

using Clock = std::chrono::steady_clock;

std::atomic<U64>& OpenDevices() {
    static std::atomic<U64> count{0};
    return count;
}

std::mutex& TuningMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<lime::ConnectionHandle> FindBoards() {
    try {
        return lime::ConnectionRegistry::findConnections();
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_LIMESDR] Failed to enumerate devices: {}", e.what());
    } catch (...) {
        JST_ERROR("[SDR_LIMESDR] Failed to enumerate devices.");
    }
    return {};
}

// Heuristic margin, not a measured end-to-end pipeline bound. It mirrors the
// RX batch sizing in LimeSuite 23.11 `Streamer::SetupStream()` (Streamer.cpp
// 235-243) for one RX channel with I12 packets: every queued USB transfer
// (`USB_MAX_CONTEXTS`), the latest received packet, and one more for the status
// counter race.
U64 InFlightSamples(const F64 sampleRate) {
    const F64 limit = (sampleRate / 1.0e6 + 5.0) * StreamLatency;
    U64 batch = 1;
    while (static_cast<F64>(batch * 2) < limit) {
        batch *= 2;
    }
    return PacketSamples * (batch * UsbTransfers + 2);
}

std::chrono::milliseconds Remaining(const Clock::time_point deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    return std::max(left, std::chrono::milliseconds(1));
}

}  // namespace

struct LimeSdrDevice::Impl {
    explicit Impl(LimeSdrDevice& owner) : device(owner) {}

    Result open(const Info& info);
    Result applyGain(F64 gain, bool force);
    Result setFrequency(F64 frequency, bool useCache);
    Result startStream();
    void stopStream();
    void poll();

    LimeSdrDevice& device;
    lms_device_t* handle = nullptr;
    lms_stream_t stream = {};
    bool streamReady = false;
    bool streaming = false;
    bool cachedTuning = false;
    U64 streamGeneration = 0;
    std::vector<int> antennaIndices;
    Clock::time_point received{};
    TimestampFilter timestamps;
    std::vector<CF32> drain;
    U64 lostPackets = 0;
    U64 overruns = 0;
};

LimeSdrDevice::LimeSdrDevice() : pimpl(std::make_unique<Impl>(*this)) {}

LimeSdrDevice::~LimeSdrDevice() {
    stop();
    if (pimpl->handle) {
        if (pimpl->streamReady) {
            LMS_DestroyStream(pimpl->handle, &pimpl->stream);
        }
        LMS_Close(pimpl->handle);
        --OpenDevices();
    }
}

std::vector<Device::Info> LimeSdrDevice::Enumerate() {
    std::vector<Info> devices;
    for (const auto& board : FindBoards()) {
        devices.push_back({
            .serial = board.serial,
            .label = jst::fmt::format("{} ({})", board.name.empty() ? "LimeSDR" : board.name,
                                      board.serial),
        });
    }
    return devices;
}

Result LimeSdrDevice::Create(const Info& info, std::unique_ptr<Device>& device) {
    auto lime = std::make_unique<LimeSdrDevice>();
    JST_CHECK(lime->pimpl->open(info));
    device = std::move(lime);
    return Result::SUCCESS;
}

Result LimeSdrDevice::Impl::open(const Info& info) {
    std::vector<std::string> matches;
    for (const auto& board : FindBoards()) {
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
    ++OpenDevices();
    device.deviceInfo = info;

    if (LMS_Init(handle) != 0 || LMS_EnableChannel(handle, LMS_CH_RX, 0, true) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to initialize device: {}", LMS_GetLastErrorMessage());
        return Result::ERROR;
    }

    lms_range_t frequency = {};
    lms_range_t rate = {};
    if (LMS_GetLOFrequencyRange(handle, LMS_CH_RX, &frequency) != 0 ||
        LMS_GetSampleRateRange(handle, LMS_CH_RX, &rate) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to query device ranges.");
        return Result::ERROR;
    }

    auto& caps = device.deviceCapabilities;
    caps.frequencyRanges = {{frequency.min, frequency.max}};
    caps.sampleRateRanges = {{rate.min, rate.max}};
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
    caps.sweepSampleRate = std::min(rate.max, 20.0e6);
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    device.defaultSettlingTime = SettlingTime;
    device.appliedSettings.gainRange = Range{0.0, static_cast<F64>(MaxGain)};
    return Result::SUCCESS;
}

Result LimeSdrDevice::start(const Settings& settings) {
    pimpl->stopStream();

    auto* handle = pimpl->handle;
    if (!Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_LIMESDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_LIMESDR] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
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

    {
        std::lock_guard lock(TuningMutex());
        if (LMS_SetSampleRate(handle, settings.sampleRate, 0) != 0 ||
            LMS_SetLOFrequency(handle, LMS_CH_RX, 0, settings.frequency) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to configure device: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        if (antenna != antennas.end() &&
            LMS_SetAntenna(handle, LMS_CH_RX, 0,
                           pimpl->antennaIndices[antenna - antennas.begin()]) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to select antenna: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        JST_CHECK(pimpl->applyGain(settings.gain, true));
        lms_range_t filter = {};
        if (LMS_GetLPFBWRange(handle, LMS_CH_RX, &filter) != 0 ||
            LMS_SetLPFBW(handle, LMS_CH_RX, 0,
                         std::clamp(settings.sampleRate, filter.min, filter.max)) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set the receive filter: {}",
                      LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        const F64 bandwidth = std::max(settings.sampleRate, 2.5e6);
        if (LMS_Calibrate(handle, LMS_CH_RX, 0, bandwidth, 0) != 0) {
            JST_WARN("[SDR_LIMESDR] Calibration failed: {}", LMS_GetLastErrorMessage());
        }
    }

    F64 hostRate = 0.0;
    F64 rfRate = 0.0;
    F64 frequency = 0.0;
    if (LMS_GetSampleRate(handle, LMS_CH_RX, 0, &hostRate, &rfRate) != 0 ||
        LMS_GetLOFrequency(handle, LMS_CH_RX, 0, &frequency) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read back the applied settings.");
        return Result::ERROR;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = hostRate;
    appliedSettings.frequency = frequency;
    appliedSettings.antenna = settings.antenna;
    appliedSettings.automaticGain = false;
    appliedSettings.biasTee = false;

    if (!pimpl->streamReady) {
        auto& stream = pimpl->stream;
        stream = {};
        stream.isTx = false;
        stream.channel = 0;
        stream.fifoSize = 1 << 20;
        stream.throughputVsLatency = StreamLatency;
        stream.dataFmt = lms_stream_t::LMS_FMT_F32;
        if (LMS_SetupStream(handle, &stream) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set up the stream: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        pimpl->streamReady = true;
    }
    if (settings.biasTee) {
        JST_WARN("[SDR_LIMESDR] Bias-T is not supported by LimeSDR.");
    }
    return pimpl->startStream();
}

void LimeSdrDevice::stop() {
    pimpl->stopStream();
}

Result LimeSdrDevice::setAutomaticGain(const bool automaticGain) {
    if (automaticGain) {
        JST_WARN("[SDR_LIMESDR] LimeSDR has no automatic gain control. Using the manual gain.");
    }
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result LimeSdrDevice::setGain(const F64 gain) {
    Result result = Result::SUCCESS;
    {
        std::lock_guard lock(TuningMutex());
        result = pimpl->applyGain(gain, false);
    }
    if (result != Result::SUCCESS) {
        return Device::setGain(gain);
    }
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 LimeSdrDevice::resolveGain(const F64 gain, const std::optional<Range>& range) const {
    return std::round(Device::resolveGain(gain, range));
}

Result LimeSdrDevice::Impl::applyGain(const F64 gain, const bool force) {
    const F64 target = device.resolveGain(gain, device.appliedSettings.gainRange);
    if (!force && target == device.appliedSettings.gain) {
        return Result::SUCCESS;
    }
    if (LMS_SetGaindB(handle, LMS_CH_RX, 0, static_cast<unsigned>(target)) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to set the gain to {:.0f} dB: {}", target,
                  LMS_GetLastErrorMessage());
        return Result::ERROR;
    }
    device.appliedSettings.gain = target;
    return Result::SUCCESS;
}

Result LimeSdrDevice::tune(const F64 frequency) {
    if (!Range::Contains(deviceCapabilities.frequencyRanges, frequency)) {
        JST_ERROR("[SDR_LIMESDR] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }
    const auto set = [&] {
        JST_CHECK(pimpl->setFrequency(frequency, pimpl->cachedTuning && OpenDevices() == 1));
        F64 actual = frequency;
        LMS_GetLOFrequency(pimpl->handle, LMS_CH_RX, 0, &actual);
        requestedSettings.frequency = frequency;
        appliedSettings.frequency = actual;
        return Result::SUCCESS;
    };
    const auto change = [&] {
        JST_CHECK(set());
        return pimpl->startStream();
    };
    if (pimpl->streaming && set() == Result::SUCCESS) {
        return Result::SUCCESS;
    }
    return restart(change);
}

Device::Capture LimeSdrDevice::read(std::span<CF32> samples, const std::chrono::milliseconds timeout) {
    auto& timestamps = pimpl->timestamps;
    const auto finish = [&](Capture capture) {
        capture.generation = pimpl->streamGeneration;
        capture.frequency = appliedSettings.frequency;
        capture.sampleRate = appliedSettings.sampleRate;
        capture.epoch = timestamps.epoch();
        return capture;
    };
    const auto receive = [&](CF32* data, const U64 count, lms_stream_meta_t& meta,
                             const std::chrono::milliseconds wait) {
        const int received = LMS_RecvStream(&pimpl->stream, data, count, &meta,
                                            static_cast<unsigned>(wait.count()));
        if (received > 0) {
            pimpl->received = Clock::now();
        }
        if (received >= 0) {
            pimpl->poll();
        }
        return received;
    };
    const auto idle = [&] {
        if (Clock::now() - pimpl->received > StallTimeout) {
            return finish({.status = Status::Disconnected,
                           .error = "LimeSDR stopped delivering samples"});
        }
        return finish({.status = Status::Timeout});
    };

    if (!pimpl->streaming) {
        return {.status = Status::Failure, .error = "stream is not running"};
    }

    const auto deadline = Clock::now() + timeout;
    while (const U64 pending = timestamps.pending()) {
        const U64 count = std::min<U64>(pending, pimpl->drain.size());
        lms_stream_meta_t meta = {};
        const int received = receive(pimpl->drain.data(), count, meta, Remaining(deadline));
        if (received < 0) {
            return finish({.status = Status::Disconnected, .error = LMS_GetLastErrorMessage()});
        }
        if (received == 0) {
            return idle();
        }
        const auto decision = timestamps.accept(meta.timestamp, static_cast<U64>(received));
        if (decision.status == Status::Failure || decision.status == Status::Discontinuity) {
            return finish({.status = decision.status, .lost = decision.lost,
                           .error = decision.error});
        }
        if (Clock::now() >= deadline) {
            return finish({.status = Status::Timeout});
        }
    }

    lms_stream_meta_t meta = {};
    const int received = receive(samples.data(), samples.size(), meta, Remaining(deadline));
    if (received < 0) {
        return finish({.status = Status::Disconnected, .error = LMS_GetLastErrorMessage()});
    }
    if (received == 0) {
        return idle();
    }

    const U64 count = static_cast<U64>(received);
    const auto decision = timestamps.accept(meta.timestamp, count);
    Capture capture{.status = decision.status, .lost = decision.lost, .error = decision.error};
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

Result LimeSdrDevice::tuneHop(const U64 hop) {
    const F64 frequency = sweepRequest.frequencies[hop];
    if (!pimpl->streaming) {
        JST_ERROR("[SDR_LIMESDR] Cannot retune a stopped stream.");
        return Result::ERROR;
    }
    JST_CHECK(pimpl->setFrequency(frequency, pimpl->cachedTuning && OpenDevices() == 1));
    F64 actual = frequency;
    LMS_GetLOFrequency(pimpl->handle, LMS_CH_RX, 0, &actual);
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = actual;
    pimpl->streamGeneration = nextGeneration();

    lms_stream_status_t status = {};
    if (LMS_GetStreamStatus(&pimpl->stream, &status) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read the stream position after retuning.");
        return Result::ERROR;
    }
    const U64 margin = InFlightSamples(appliedSettings.sampleRate);
    if (guardLength > std::numeric_limits<U64>::max() - margin ||
        status.timestamp > std::numeric_limits<U64>::max() - margin - guardLength) {
        JST_ERROR("[SDR_LIMESDR] Settling guard exceeds the hardware timestamp range.");
        return Result::ERROR;
    }
    pimpl->timestamps.retune(status.timestamp + margin + guardLength);
    return Result::SUCCESS;
}

void LimeSdrDevice::Impl::poll() {
    lms_stream_status_t status = {};
    if (LMS_GetStreamStatus(&stream, &status) != 0) {
        return;
    }
    lostPackets += status.droppedPackets;
    overruns += status.overrun;
    if (status.droppedPackets == 0 && status.overrun == 0) {
        return;
    }
    const U64 margin = 2 * PacketSamples;
    timestamps.lose(status.timestamp > std::numeric_limits<U64>::max() - margin
        ? std::numeric_limits<U64>::max()
        : status.timestamp + margin);
}

Result LimeSdrDevice::prepareSweep(const SweepRequest& request) {
    pimpl->cachedTuning = false;
    JST_CHECK(Device::prepareSweep(request));
    if (OpenDevices() != 1) {
        JST_WARN("[SDR_LIMESDR] Several LimeSDR devices are open. Cached tuning maps are "
                 "shared across the process, so full tuning is used.");
        return Result::SUCCESS;
    }
    pimpl->cachedTuning = true;
    return Result::SUCCESS;
}

Result LimeSdrDevice::Impl::setFrequency(const F64 frequency, const bool useCache) {
    bool tuned = false;
    {
        std::lock_guard lock(TuningMutex());
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

Result LimeSdrDevice::Impl::startStream() {
    device.guardLength = device.guardSamples(device.requestedSettings,
                                             device.appliedSettings.sampleRate);
    if (device.guardLength == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR_LIMESDR] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    try {
        drain.resize(DrainSamples);
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_LIMESDR] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    if (LMS_StartStream(&stream) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to start the stream: {}", LMS_GetLastErrorMessage());
        return Result::ERROR;
    }
    lms_stream_status_t status = {};
    LMS_GetStreamStatus(&stream, &status);
    streamGeneration = device.nextGeneration();
    received = Clock::now();
    timestamps.restart(device.guardLength, static_cast<U64>(device.appliedSettings.sampleRate));
    streaming = true;
    return Result::SUCCESS;
}

void LimeSdrDevice::Impl::stopStream() {
    if (streaming) {
        LMS_StopStream(&stream);
    }
    streaming = false;
}

}  // namespace Jetstream::Radio
