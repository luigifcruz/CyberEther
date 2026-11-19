#include "base.hh"
#include "tools/library.hh"

#include "../stream.hh"
#include "../../tools/timestamp_filter.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
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
constexpr F64 MinMultiChannelFrequency = 30.0e6;
constexpr int MaxReceiveChannels = 2;

std::chrono::milliseconds Remaining(const Clock::time_point deadline) {
    const auto left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    return std::max(left, std::chrono::milliseconds(1));
}

class LimeSdrStream final : public Stream {
 public:
    ~LimeSdrStream() override;

    Result connect(const Device::Info& info);
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;
    Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;
    Device::Capture readChannels(std::span<std::span<CF32>> samples,
                                 std::chrono::milliseconds timeout) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(F64 frequency) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;
    Result setChannelGains(const std::vector<F64>& gains) override;

 private:
    enum class Delivery {
        Samples,
        Empty,
        Misaligned,
        Failed,
    };

    void disconnect();
    void destroyStreams();
    static Result CheckFrequency(U64 channels, F64 frequency);
    Result setOscillator(F64 frequency, bool useCache);
    Result tuneOscillator(F64 frequency, bool useCache);
    bool poll();
    Result applyGain(F64 gain, bool force);
    int receive(U64 channel, CF32* data, U64 count, lms_stream_meta_t& meta,
                std::chrono::milliseconds wait);
    Delivery receiveAll(std::span<CF32* const> data, U64 count, lms_stream_meta_t& meta,
                        Clock::time_point deadline, U64& delivered);
    Delivery align(Clock::time_point deadline);

    lms_device_t* handle = nullptr;
    std::vector<int> antennaIndices;
    Device::Range tunableRange;
    std::vector<lms_stream_t> streams;
    U64 streamBatch = 1;
    bool streaming = false;
    bool aligned = false;
    U64 receiveChannels = 1;
    std::vector<F64> channelGains;
    U64 queueGeneration = 0;
    Clock::time_point received{};
    TimestampFilter timestamps;
    std::vector<std::vector<CF32>> drains;
    std::vector<CF32*> drainPointers;
    std::vector<CF32*> outputs;
};

LimeSdrStream::~LimeSdrStream() {
    disconnect();
}

void LimeSdrStream::destroyStreams() {
    for (auto& stream : streams) {
        LMS_DestroyStream(handle, &stream);
    }
    streams.clear();
}

void LimeSdrStream::disconnect() {
    close();
    if (handle) {
        destroyStreams();
        LMS_Close(handle);
        handle = nullptr;
        --LimeSdrLibrary::OpenDevices();
    }
}

Result LimeSdrStream::connect(const Device::Info& info) {
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
    caps.receiveChannels = static_cast<U64>(
        std::clamp(LMS_GetNumChannels(handle, LMS_CH_RX), 1, MaxReceiveChannels));
    tunableRange = {frequency.min, frequency.max};
    caps.frequencyRanges = {tunableRange};
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

Result LimeSdrStream::CheckFrequency(const U64 channels, const F64 frequency) {
    if (channels > 1 && frequency < MinMultiChannelFrequency) {
        JST_ERROR("[SDR_LIMESDR] Multiple RX channels need a frequency of {:.0f} MHz or more, "
                  "and {:.3f} MHz was requested. Use Channels 1 below {:.0f} MHz.",
                  MinMultiChannelFrequency / 1e6, frequency / 1e6,
                  MinMultiChannelFrequency / 1e6);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result LimeSdrStream::apply(const Settings& settings) {
    if (!streams.empty()) {
        disconnect();
    }
    if (!handle) {
        const auto info = deviceInfo;
        if (const auto result = connect(info); result != Result::SUCCESS) {
            disconnect();
            return result;
        }
    }
    deviceCapabilities.frequencyRanges = {{
        settings.receiveChannels > 1 ? std::max(tunableRange.minimum, MinMultiChannelFrequency)
                                     : tunableRange.minimum,
        tunableRange.maximum,
    }};
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_LIMESDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    JST_CHECK(CheckFrequency(settings.receiveChannels, settings.frequency));
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_LIMESDR] Frequency ({:.3f} MHz) not supported. Use {}.",
                  settings.frequency / 1e6,
                  Device::Range::Describe(deviceCapabilities.frequencyRanges));
        return Result::ERROR;
    }
    const auto& antennas = deviceCapabilities.antennas;
    std::vector<std::optional<int>> antennaPaths(settings.receiveChannels);
    for (U64 i = 0; i < settings.receiveChannels; ++i) {
        const auto& name = i < settings.channelAntennas.size() ? settings.channelAntennas[i]
                                                               : settings.antenna;
        if (name.empty()) {
            continue;
        }
        const auto antenna = std::find(antennas.begin(), antennas.end(), name);
        if (antenna == antennas.end()) {
            JST_ERROR("[SDR_LIMESDR] Receive antenna '{}' is not supported.", name);
            return Result::ERROR;
        }
        antennaPaths[i] = antennaIndices[antenna - antennas.begin()];
    }
    if (settings.automaticGain) {
        JST_WARN("[SDR_LIMESDR] LimeSDR has no automatic gain control. Using the manual gain.");
    }
    if (settings.biasTee) {
        JST_WARN("[SDR_LIMESDR] Bias-T is not supported by LimeSDR.");
    }

    receiveChannels = settings.receiveChannels;
    channelGains = settings.channelGains;
    for (U64 i = 0; i < deviceCapabilities.receiveChannels; ++i) {
        if (LMS_EnableChannel(handle, LMS_CH_RX, i, i < receiveChannels) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to {} RX{}: {}", i < receiveChannels ? "enable"
                                                                                : "disable",
                      i + 1, LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
    }

    {
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        if (LMS_SetSampleRate(handle, settings.sampleRate, 0) != 0 ||
            LMS_SetLOFrequency(handle, LMS_CH_RX, 0, settings.frequency) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to configure device: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        for (U64 i = 0; i < antennaPaths.size(); ++i) {
            if (antennaPaths[i] &&
                LMS_SetAntenna(handle, LMS_CH_RX, i, static_cast<size_t>(*antennaPaths[i])) != 0) {
                JST_ERROR("[SDR_LIMESDR] Failed to select the RX{} antenna: {}", i + 1,
                          LMS_GetLastErrorMessage());
                return Result::ERROR;
            }
        }
        JST_CHECK(applyGain(settings.gain, true));
        lms_range_t filter = {};
        if (LMS_GetLPFBWRange(handle, LMS_CH_RX, &filter) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set the receive filter: {}",
                      LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
        const F64 bandwidth = std::max(settings.sampleRate, MinCalibrationBandwidth);
        for (U64 i = 0; i < receiveChannels; ++i) {
            if (LMS_SetLPFBW(handle, LMS_CH_RX, i,
                             std::clamp(settings.sampleRate, filter.min, filter.max)) != 0) {
                JST_ERROR("[SDR_LIMESDR] Failed to set the receive filter: {}",
                          LMS_GetLastErrorMessage());
                return Result::ERROR;
            }
            if (LMS_Calibrate(handle, LMS_CH_RX, i, bandwidth, 0) != 0) {
                JST_WARN("[SDR_LIMESDR] Failed to calibrate RX{}: {}", i + 1,
                         LMS_GetLastErrorMessage());
            }
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
    return Result::SUCCESS;
}

Result LimeSdrStream::setOscillator(const F64 frequency, const bool useCache) {
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

Result LimeSdrStream::tuneOscillator(const F64 frequency, const bool useCache) {
    JST_CHECK(setOscillator(frequency, useCache));
    F64 actual = 0.0;
    if (LMS_GetLOFrequency(handle, LMS_CH_RX, 0, &actual) != 0) {
        JST_ERROR("[SDR_LIMESDR] Failed to read back the tuned frequency: {}",
                  LMS_GetLastErrorMessage());
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = actual;
    return Result::SUCCESS;
}

Result LimeSdrStream::open() {
    const F64 sampleRate = appliedSettings.sampleRate;
    try {
        drains.assign(receiveChannels, std::vector<CF32>(DrainSamples));
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_LIMESDR] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    drainPointers.clear();
    for (auto& drain : drains) {
        drainPointers.push_back(drain.data());
    }
    if (streams.empty()) {
        streams.reserve(receiveChannels);
        for (U64 i = 0; i < receiveChannels; ++i) {
            lms_stream_t stream = {};
            stream.isTx = false;
            stream.channel = static_cast<uint32_t>(i);
            stream.fifoSize = FifoSamples;
            stream.throughputVsLatency = Latency;
            stream.dataFmt = lms_stream_t::LMS_FMT_F32;
            if (LMS_SetupStream(handle, &stream) != 0) {
                JST_ERROR("[SDR_LIMESDR] Failed to set up the stream: {}",
                          LMS_GetLastErrorMessage());
                destroyStreams();
                return Result::ERROR;
            }
            streams.push_back(stream);
        }
        const F64 limit = (sampleRate / 1.0e6 + 5.0) * Latency;
        for (U64 batch = 1; static_cast<F64>(batch) < limit; batch *= 2) {
            streamBatch = batch;
        }
    }
    for (U64 i = 0; i < streams.size(); ++i) {
        if (LMS_StartStream(&streams[i]) != 0) {
            while (i > 0) {
                LMS_StopStream(&streams[--i]);
            }
            JST_ERROR("[SDR_LIMESDR] Failed to start the stream: {}", LMS_GetLastErrorMessage());
            return Result::ERROR;
        }
    }
    lms_stream_status_t status = {};
    if (LMS_GetStreamStatus(&streams.front(), &status) != 0) {
        for (auto& stream : streams) {
            LMS_StopStream(&stream);
        }
        JST_ERROR("[SDR_LIMESDR] Failed to read the stream status.");
        return Result::ERROR;
    }
    queueGeneration = nextGeneration();
    received = Clock::now();
    timestamps.restart(guard(), static_cast<U64>(sampleRate));
    aligned = streams.size() == 1;
    streaming = true;
    return Result::SUCCESS;
}

void LimeSdrStream::close() {
    if (streaming) {
        for (auto& stream : streams) {
            LMS_StopStream(&stream);
        }
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
    if (LMS_GetStreamStatus(&streams.front(), &status) != 0) {
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
    bool lost = false;
    U64 latest = 0;
    for (auto& stream : streams) {
        lms_stream_status_t status = {};
        if (LMS_GetStreamStatus(&stream, &status) != 0) {
            return false;
        }
        if (status.droppedPackets != 0 || status.overrun != 0) {
            lost = true;
            latest = std::max<U64>(latest, status.timestamp);
        }
    }
    if (!lost) {
        return true;
    }
    const U64 margin = 2 * PacketSamples;
    timestamps.lose(latest > std::numeric_limits<U64>::max() - margin
        ? std::numeric_limits<U64>::max()
        : latest + margin);
    return true;
}

int LimeSdrStream::receive(const U64 channel, CF32* data, const U64 count,
                           lms_stream_meta_t& meta, const std::chrono::milliseconds wait) {
    return LMS_RecvStream(&streams[channel], data, count, &meta,
                          static_cast<unsigned>(wait.count()));
}

LimeSdrStream::Delivery LimeSdrStream::receiveAll(std::span<CF32* const> data, const U64 count,
                                                  lms_stream_meta_t& meta,
                                                  const Clock::time_point deadline,
                                                  U64& delivered) {
    delivered = 0;
    const int first = receive(0, data[0], count, meta, Remaining(deadline));
    if (first < 0) {
        return Delivery::Failed;
    }
    if (first == 0) {
        return Delivery::Empty;
    }
    delivered = static_cast<U64>(first);
    for (U64 channel = 1; channel < streams.size(); ++channel) {
        U64 matched = 0;
        while (matched < delivered) {
            lms_stream_meta_t other = {};
            const int read = receive(channel, data[channel] + matched, delivered - matched, other,
                                     Remaining(deadline));
            if (read < 0) {
                return Delivery::Failed;
            }
            if (read == 0 && Clock::now() < deadline) {
                continue;
            }
            if (read == 0 || other.timestamp != meta.timestamp + matched) {
                aligned = false;
                return Delivery::Misaligned;
            }
            matched += static_cast<U64>(read);
        }
    }
    received = Clock::now();
    return Delivery::Samples;
}

LimeSdrStream::Delivery LimeSdrStream::align(const Clock::time_point deadline) {
    std::vector<U64> next(streams.size());
    for (U64 channel = 0; channel < streams.size(); ++channel) {
        lms_stream_meta_t meta = {};
        const int delivered = receive(channel, drains[channel].data(), PacketSamples, meta,
                                      Remaining(deadline));
        if (delivered < 0) {
            return Delivery::Failed;
        }
        if (delivered == 0) {
            return Delivery::Empty;
        }
        next[channel] = meta.timestamp + static_cast<U64>(delivered);
    }
    while (true) {
        const U64 target = *std::max_element(next.begin(), next.end());
        if (std::all_of(next.begin(), next.end(), [&](const U64 value) {
                return value == target;
            })) {
            aligned = true;
            return Delivery::Samples;
        }
        if (Clock::now() >= deadline) {
            return Delivery::Empty;
        }
        for (U64 channel = 0; channel < streams.size(); ++channel) {
            if (next[channel] == target) {
                continue;
            }
            const U64 count = std::min<U64>(target - next[channel], drains[channel].size());
            lms_stream_meta_t meta = {};
            const int delivered = receive(channel, drains[channel].data(), count, meta,
                                          Remaining(deadline));
            if (delivered < 0) {
                return Delivery::Failed;
            }
            if (delivered == 0) {
                return Delivery::Empty;
            }
            next[channel] = meta.timestamp + static_cast<U64>(delivered);
        }
    }
}

Device::Capture LimeSdrStream::read(std::span<CF32> samples,
                                    const std::chrono::milliseconds timeout) {
    std::array destinations{samples};
    return readChannels(destinations, timeout);
}

Device::Capture LimeSdrStream::readChannels(std::span<std::span<CF32>> samples,
                                            const std::chrono::milliseconds timeout) {
    const auto finish = [&](Device::Capture capture) {
        capture.generation = queueGeneration;
        capture.frequency = appliedSettings.frequency;
        capture.sampleRate = appliedSettings.sampleRate;
        capture.epoch = timestamps.epoch();
        return capture;
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
    const auto failed = [&] {
        return finish({.status = Status::Disconnected, .error = LMS_GetLastErrorMessage()});
    };

    if (!streaming) {
        return {.status = Status::Failure, .error = "stream is not running"};
    }
    if (samples.size() != streams.size() || samples.front().empty() ||
        std::any_of(samples.begin(), samples.end(), [&](const auto channel) {
            return channel.size() != samples.front().size();
        })) {
        return {.status = Status::Failure, .error = "invalid receive channel buffers"};
    }
    const U64 size = samples.front().size();

    const auto deadline = Clock::now() + timeout;
    if (!aligned) {
        const auto delivery = align(deadline);
        if (delivery == Delivery::Failed) {
            return failed();
        }
        if (delivery == Delivery::Empty) {
            return idle();
        }
    }

    while (const U64 pending = timestamps.pending()) {
        const U64 count = std::min<U64>(pending, drains.front().size());
        lms_stream_meta_t meta = {};
        U64 delivered = 0;
        const auto delivery = receiveAll(drainPointers, count, meta, deadline, delivered);
        if (delivery == Delivery::Failed) {
            return failed();
        }
        if (!poll()) {
            return blind();
        }
        if (delivery == Delivery::Empty) {
            return idle();
        }
        if (delivery == Delivery::Misaligned) {
            return idle();
        }
        const auto decision = timestamps.accept(meta.timestamp, delivered);
        if (decision.status == Status::Failure || decision.status == Status::Discontinuity) {
            return finish({.status = decision.status, .lost = decision.lost,
                           .error = decision.error});
        }
        if (Clock::now() >= deadline) {
            return finish({.status = Status::Timeout});
        }
    }

    outputs.clear();
    for (const auto channel : samples) {
        outputs.push_back(channel.data());
    }
    lms_stream_meta_t meta = {};
    U64 count = 0;
    const auto delivery = receiveAll(outputs, size, meta, deadline, count);
    if (delivery == Delivery::Failed) {
        return failed();
    }
    if (!poll()) {
        return blind();
    }
    if (delivery == Delivery::Empty) {
        return idle();
    }
    if (delivery == Delivery::Misaligned) {
        return idle();
    }

    const auto decision = timestamps.accept(meta.timestamp, count);
    Device::Capture capture{.status = decision.status, .lost = decision.lost,
                            .error = decision.error};
    if (decision.status != Status::Samples) {
        return finish(capture);
    }
    if (decision.skip > 0) {
        for (const auto channel : samples) {
            std::copy(channel.begin() + decision.skip, channel.begin() + count, channel.begin());
        }
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
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        JST_CHECK(applyGain(gain, false));
    }
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

Result LimeSdrStream::setChannelGains(const std::vector<F64>& gains) {
    channelGains = gains;
    {
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        JST_CHECK(applyGain(requestedSettings.gain, false));
    }
    requestedSettings.channelGains = gains;
    return Result::SUCCESS;
}

F64 LimeSdrStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range));
}

Result LimeSdrStream::applyGain(const F64 gain, const bool force) {
    appliedSettings.channelGains.resize(receiveChannels, std::numeric_limits<F64>::quiet_NaN());
    for (U64 i = 0; i < receiveChannels; ++i) {
        const F64 requested = i < channelGains.size() ? channelGains[i] : gain;
        const F64 target = resolveGain(requested, appliedSettings.gainRange);
        if (!force && target == appliedSettings.channelGains[i]) {
            continue;
        }
        if (LMS_SetGaindB(handle, LMS_CH_RX, i, static_cast<unsigned>(target)) != 0) {
            JST_ERROR("[SDR_LIMESDR] Failed to set the RX{} gain to {:.0f} dB: {}", i + 1,
                      target, LMS_GetLastErrorMessage());
            appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
            appliedSettings.channelGains[i] = std::numeric_limits<F64>::quiet_NaN();
            return Result::ERROR;
        }
        appliedSettings.channelGains[i] = target;
    }
    appliedSettings.gain = resolveGain(gain, appliedSettings.gainRange);
    return Result::SUCCESS;
}

}  // namespace

Result LimeSdr::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<LimeSdrStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
