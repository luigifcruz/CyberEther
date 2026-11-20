#include "base.hh"
#include "tools/library.hh"

#include "../stream.hh"
#include "../../tools/timestamp_filter.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <vector>

#include <limesuiteng/DeviceRegistry.h>
#include <limesuiteng/Logger.h>
#include <limesuiteng/RFStream.h>
#include <limesuiteng/SDRDescriptor.h>
#include <limesuiteng/SDRDevice.h>
#include <limesuiteng/StreamConfig.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

using Status = Device::Status;
using Clock = std::chrono::steady_clock;

constexpr F32 Latency = 0.5f;
constexpr F64 SettlingTime = 2.0e-3;
constexpr U64 PacketSamples = 1360;
constexpr U64 DrainSamples = 65536;
constexpr U64 UsbTransfers = 16;
constexpr U32 FifoSamples = 1 << 20;
constexpr F64 SweepSampleRate = 20.0e6;
constexpr F64 MinCalibrationBandwidth = 2.5e6;
constexpr F64 MinMultiChannelFrequency = 30.0e6;
constexpr F64 MinOscillatorFrequency = 30.0e6;
constexpr F64 AutoAntennaSplit = 1.7e9;
constexpr U8 AutoAntenna = std::numeric_limits<U8>::max();
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
    Device::Capture read(std::span<std::byte> samples, std::chrono::milliseconds timeout) override;
    Device::Capture readChannels(std::span<std::span<std::byte>> samples,
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
    void disconnect();
    static Result CheckFrequency(U64 channels, F64 frequency);
    Result setCarrier(F64 frequency);
    Result setOscillator(F64 frequency, bool useCache);
    Result tuneOscillator(F64 frequency, bool useCache);
    void poll();
    Result applyGain(F64 gain, bool force);
    U64 receive(std::span<std::byte* const> data, U64 count, lime::StreamMeta& meta,
                std::chrono::milliseconds wait);

    lime::SDRDevice* device = nullptr;
    std::vector<U8> antennaIndices;
    std::vector<std::optional<U8>> channelAntennas;
    std::optional<U8> autoLowAntenna;
    std::optional<U8> autoHighAntenna;
    bool sharedAntenna = false;
    F64 mixerOffset = std::numeric_limits<F64>::quiet_NaN();
    Device::Range tunableRange;
    std::unique_ptr<lime::RFStream> stream;
    DataType streamFormat = DataType::CF32;
    F64 gainOffset = 0.0;
    U64 streamBatch = 1;
    U64 streamLoss = 0;
    bool streaming = false;
    U64 receiveChannels = 1;
    std::vector<F64> channelGains;
    U64 queueGeneration = 0;
    Clock::time_point received{};
    TimestampFilter timestamps;
    std::vector<std::vector<std::byte>> drains;
    std::vector<std::byte*> drainPointers;
    std::vector<std::byte*> outputs;
};

LimeSdrStream::~LimeSdrStream() {
    disconnect();
}

void LimeSdrStream::disconnect() {
    close();
    if (device) {
        stream.reset();
        lime::DeviceRegistry::freeDevice(device);
        device = nullptr;
        --LimeSdrLibrary::OpenDevices();
    }
}

Result LimeSdrStream::connect(const Device::Info& info) {
    std::vector<lime::DeviceHandle> matches;
    for (const auto& board : LimeSdrLibrary::Boards()) {
        if (!board.serial.empty() && board.serial == info.serial) {
            matches.push_back(board);
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
    try {
        device = lime::DeviceRegistry::makeDevice(match);
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_LIMESDR] Failed to open device '{}': {}", info.label, e.what());
        return Result::ERROR;
    }
    if (!device) {
        JST_ERROR("[SDR_LIMESDR] Failed to open device '{}': {}", info.label,
                  lime::GetLastErrorMessage());
        return Result::ERROR;
    }
    ++LimeSdrLibrary::OpenDevices();
    deviceInfo = info;

    try {
        if (device->Init() != lime::OpStatus::Success ||
            device->EnableChannel(0, lime::TRXDir::Rx, 0, true) != lime::OpStatus::Success) {
            JST_ERROR("[SDR_LIMESDR] Failed to initialize the device: {}",
                      lime::GetLastErrorMessage());
            return Result::ERROR;
        }
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_LIMESDR] Failed to initialize the device: {}", e.what());
        return Result::ERROR;
    }

    const auto& descriptor = device->GetDescriptor();
    if (descriptor.rfSOC.empty()) {
        JST_ERROR("[SDR_LIMESDR] Failed to query device ranges.");
        return Result::ERROR;
    }
    const auto& chip = descriptor.rfSOC.front();
    const auto& rate = chip.samplingRateRange;
    const auto gains = chip.gainRange.find(lime::TRXDir::Rx);
    if (gains == chip.gainRange.end() || !gains->second.contains(lime::eGainTypes::GENERIC)) {
        JST_ERROR("[SDR_LIMESDR] Failed to query the receive gain range.");
        return Result::ERROR;
    }
    const auto& gain = gains->second.at(lime::eGainTypes::GENERIC);
    gainOffset = gain.min;

    auto& caps = deviceCapabilities;
    caps.receiveChannels = static_cast<U64>(
        std::clamp(static_cast<int>(chip.channelCount), 1, MaxReceiveChannels));
    tunableRange = {chip.frequencyRange.min, chip.frequencyRange.max};
    caps.frequencyRanges = {tunableRange};
    caps.sampleRateRanges = {{rate.min, rate.max}};
    caps.bandwidthRanges.clear();
    if (const auto filter = chip.lowPassFilterRange.find(lime::TRXDir::Rx);
        filter != chip.lowPassFilterRange.end()) {
        caps.bandwidthRanges = {{filter->second.min, filter->second.max}};
    }
    caps.formats = {{}, {DataType::CI16, 2048.0}};
    caps.antennas.clear();
    antennaIndices.clear();
    autoLowAntenna.reset();
    autoHighAntenna.reset();
    bool autoAntenna = false;
    if (const auto paths = chip.pathNames.find(lime::TRXDir::Rx); paths != chip.pathNames.end()) {
        for (U64 i = 0; i < paths->second.size(); ++i) {
            const auto& name = paths->second[i];
            if (name == "NONE" || name == "None") {
                continue;
            }
            if (name == "LNAH") {
                autoHighAntenna = static_cast<U8>(i);
            } else if (name == "LNAW" || name == "LNAL") {
                autoLowAntenna = autoLowAntenna.value_or(static_cast<U8>(i));
            } else if (name.ends_with("_NC")) {
                autoAntenna = true;
            }
            caps.antennas.push_back(name);
            antennaIndices.push_back(static_cast<U8>(i));
        }
    }
    if (autoAntenna && autoLowAntenna && autoHighAntenna) {
        caps.antennas.push_back("Auto");
        antennaIndices.push_back(AutoAntenna);
    } else {
        autoLowAntenna.reset();
        autoHighAntenna.reset();
    }
    sharedAntenna = descriptor.name.find("XTRX") != std::string::npos;
    mixerOffset = std::numeric_limits<F64>::quiet_NaN();
    caps.automaticGain = false;
    caps.biasTee = false;
    caps.sweepSampleRate = std::min(rate.max, SweepSampleRate);
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    defaultSettlingTime = SettlingTime;
    appliedSettings.gainRange = Device::Range{0.0, gain.max - gain.min};

    JST_DEBUG("[SDR_LIMESDR] Opened '{}' at {}.", info.label, match.Serialize());
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
    if (stream) {
        disconnect();
    }
    if (!device) {
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
    std::vector<std::optional<U8>> antennaPaths(settings.receiveChannels);
    for (U64 i = 0; i < settings.receiveChannels; ++i) {
        const auto& name = i < settings.channelAntennas.size() ? settings.channelAntennas[i]
                                                               : settings.antenna;
        if (name.empty()) {
            if (autoLowAntenna) {
                antennaPaths[i] = AutoAntenna;
            }
            continue;
        }
        const auto antenna = std::find(antennas.begin(), antennas.end(), name);
        if (antenna == antennas.end()) {
            JST_ERROR("[SDR_LIMESDR] Receive antenna '{}' is not supported.", name);
            return Result::ERROR;
        }
        antennaPaths[i] = antennaIndices[antenna - antennas.begin()];
    }
    std::optional<U8> switchPath;
    bool switchConflict = false;
    for (const auto& path : antennaPaths) {
        if (path) {
            switchConflict = switchConflict || (switchPath && *switchPath != *path);
            switchPath = path;
        }
    }
    if (sharedAntenna && switchConflict) {
        JST_ERROR("[SDR_LIMESDR] LimeSDR XTRX routes every receiver through one antenna "
                  "switch. Select the same Antenna on each receiver.");
        return Result::ERROR;
    }
    if (sharedAntenna) {
        if (!switchPath) {
            switchPath = device->GetAntenna(0, lime::TRXDir::Rx, 0);
        }
        std::fill(antennaPaths.begin(), antennaPaths.end(), switchPath);
    }
    if (settings.automaticGain) {
        JST_WARN("[SDR_LIMESDR] LimeSDR has no automatic gain control. Using the manual gain.");
    }
    if (settings.biasTee) {
        JST_WARN("[SDR_LIMESDR] Bias-T is not supported by LimeSDR.");
    }

    receiveChannels = settings.receiveChannels;
    channelGains = settings.channelGains;
    channelAntennas.clear();
    for (U64 i = 0; i < deviceCapabilities.receiveChannels; ++i) {
        if (device->EnableChannel(0, lime::TRXDir::Rx, static_cast<U8>(i), i < receiveChannels) !=
            lime::OpStatus::Success) {
            JST_ERROR("[SDR_LIMESDR] Failed to {} RX{}: {}", i < receiveChannels ? "enable"
                                                                                : "disable",
                      i + 1, lime::GetLastErrorMessage());
            return Result::ERROR;
        }
    }

    {
        std::lock_guard lock(LimeSdrLibrary::TuningMutex());
        if (device->SetSampleRate(0, lime::TRXDir::Rx, 0, settings.sampleRate, 0) !=
            lime::OpStatus::Success) {
            JST_ERROR("[SDR_LIMESDR] Failed to configure device: {}",
                      lime::GetLastErrorMessage());
            return Result::ERROR;
        }
        for (U64 i = 0; i < antennaPaths.size(); ++i) {
            if (antennaPaths[i] && *antennaPaths[i] != AutoAntenna &&
                device->SetAntenna(0, lime::TRXDir::Rx, static_cast<U8>(i), *antennaPaths[i]) !=
                    lime::OpStatus::Success) {
                JST_ERROR("[SDR_LIMESDR] Failed to select the RX{} antenna: {}", i + 1,
                          lime::GetLastErrorMessage());
                return Result::ERROR;
            }
        }
        channelAntennas = antennaPaths;
        mixerOffset = std::numeric_limits<F64>::quiet_NaN();
        JST_CHECK(setCarrier(settings.frequency));
        JST_CHECK(applyGain(settings.gain, true));
        const F64 requested = settings.bandwidth > 0.0 ? settings.bandwidth : settings.sampleRate;
        const auto& filter = deviceCapabilities.bandwidthRanges;
        const F64 width = filter.empty()
            ? requested
            : std::clamp(requested, filter.front().minimum, filter.front().maximum);
        const F64 bandwidth = std::max(width, MinCalibrationBandwidth);
        for (U64 i = 0; i < receiveChannels; ++i) {
            const auto channel = static_cast<U8>(i);
            if (device->SetLowPassFilter(0, lime::TRXDir::Rx, channel, width) !=
                lime::OpStatus::Success) {
                JST_ERROR("[SDR_LIMESDR] Failed to set the receive filter: {}",
                          lime::GetLastErrorMessage());
                return Result::ERROR;
            }
            if (device->Calibrate(0, lime::TRXDir::Rx, channel, bandwidth) !=
                lime::OpStatus::Success) {
                JST_WARN("[SDR_LIMESDR] Failed to calibrate RX{}: {}", i + 1,
                         lime::GetLastErrorMessage());
            }
        }
        JST_CHECK(applyGain(settings.gain, true));
    }

    const F64 hostRate = device->GetSampleRate(0, lime::TRXDir::Rx, 0);
    const F64 frequency = device->GetFrequency(0, lime::TRXDir::Rx, 0) - mixerOffset;
    if (hostRate <= 0.0) {
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

Result LimeSdrStream::setCarrier(const F64 frequency) {
    const F64 oscillator = std::max(frequency, MinOscillatorFrequency);
    const F64 offset = oscillator - frequency;
    if (device->SetFrequency(0, lime::TRXDir::Rx, 0, oscillator) != lime::OpStatus::Success) {
        JST_ERROR("[SDR_LIMESDR] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                  lime::GetLastErrorMessage());
        return Result::ERROR;
    }
    if (offset > 0.0) {
        U32 rfRate = 0;
        const F64 hostRate = device->GetSampleRate(0, lime::TRXDir::Rx, 0, &rfRate);
        if (offset + hostRate / 2.0 >= rfRate / 2.0) {
            JST_ERROR("[SDR_LIMESDR] Frequency ({:.3f} MHz) is out of reach of the receive mixer "
                      "at {:.3f} MS/s. Raise the frequency or lower the sample rate.",
                      frequency / 1e6, hostRate / 1e6);
            return Result::ERROR;
        }
    }
    if (offset != mixerOffset) {
        for (U64 i = 0; i < receiveChannels; ++i) {
            if (device->SetNCOFrequency(0, lime::TRXDir::Rx, static_cast<U8>(i), 0, offset) !=
                lime::OpStatus::Success) {
                JST_ERROR("[SDR_LIMESDR] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                          lime::GetLastErrorMessage());
                return Result::ERROR;
            }
        }
        mixerOffset = offset;
    }
    for (U64 i = 0; i < channelAntennas.size(); ++i) {
        if (channelAntennas[i] != AutoAntenna) {
            continue;
        }
        const U8 path = frequency < AutoAntennaSplit ? *autoLowAntenna : *autoHighAntenna;
        if (device->GetAntenna(0, lime::TRXDir::Rx, static_cast<U8>(i)) != path &&
            device->SetAntenna(0, lime::TRXDir::Rx, static_cast<U8>(i), path) !=
                lime::OpStatus::Success) {
            JST_ERROR("[SDR_LIMESDR] Failed to select the RX{} antenna: {}", i + 1,
                      lime::GetLastErrorMessage());
            return Result::ERROR;
        }
    }
    return Result::SUCCESS;
}

Result LimeSdrStream::setOscillator(const F64 frequency, const bool useCache) {
    std::lock_guard lock(LimeSdrLibrary::TuningMutex());
    if (useCache) {
        device->EnableCache(true);
    }
    const auto result = setCarrier(frequency);
    if (useCache) {
        device->EnableCache(false);
    }
    return result;
}

Result LimeSdrStream::tuneOscillator(const F64 frequency, const bool useCache) {
    JST_CHECK(setOscillator(frequency, useCache));
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = device->GetFrequency(0, lime::TRXDir::Rx, 0) - mixerOffset;
    return Result::SUCCESS;
}

Result LimeSdrStream::open() {
    const F64 sampleRate = appliedSettings.sampleRate;
    try {
        drains.assign(receiveChannels, std::vector<std::byte>(DrainSamples * sampleBytes()));
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_LIMESDR] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    drainPointers.clear();
    for (auto& drain : drains) {
        drainPointers.push_back(drain.data());
    }
    if (stream && streamFormat != requestedSettings.format) {
        stream.reset();
    }
    if (!stream) {
        streamFormat = requestedSettings.format;
        const bool integer = streamFormat == DataType::CI16;
        lime::StreamConfig config;
        for (U64 i = 0; i < receiveChannels; ++i) {
            config.channels[lime::TRXDir::Rx].push_back(static_cast<U8>(i));
        }
        config.format = integer ? lime::DataFormat::I12 : lime::DataFormat::F32;
        config.linkFormat = integer ? lime::DataFormat::I12 : lime::DataFormat::I16;
        config.bufferSize = FifoSamples;
        config.hintSampleRate = static_cast<float>(sampleRate);
        try {
            stream = device->StreamCreate(config, 0);
        } catch (const std::exception& e) {
            JST_ERROR("[SDR_LIMESDR] Failed to set up the stream: {}", e.what());
            return Result::ERROR;
        }
        if (!stream) {
            JST_ERROR("[SDR_LIMESDR] Failed to set up the stream: {}",
                      lime::GetLastErrorMessage());
            return Result::ERROR;
        }
        const F64 limit = (sampleRate / 1.0e6 + 5.0) * Latency;
        for (U64 batch = 1; static_cast<F64>(batch) < limit; batch *= 2) {
            streamBatch = batch;
        }
    }
    if (stream->Start() != lime::OpStatus::Success) {
        JST_ERROR("[SDR_LIMESDR] Failed to start the stream: {}", lime::GetLastErrorMessage());
        return Result::ERROR;
    }
    lime::StreamStats status;
    stream->StreamStatus(&status, nullptr);
    streamLoss = static_cast<U64>(status.loss) + status.overrun;
    queueGeneration = nextGeneration();
    received = Clock::now();
    timestamps.restart(guard(), static_cast<U64>(sampleRate));
    streaming = true;
    return Result::SUCCESS;
}

void LimeSdrStream::close() {
    if (streaming) {
        stream->Stop();
    }
    streaming = false;
}

Result LimeSdrStream::tune(const F64 frequency) {
    JST_CHECK(tuneOscillator(frequency, false));
    if (!streaming) {
        return Result::SUCCESS;
    }
    queueGeneration = nextGeneration();

    lime::StreamStats status;
    stream->StreamStatus(&status, nullptr);
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

void LimeSdrStream::poll() {
    lime::StreamStats status;
    stream->StreamStatus(&status, nullptr);
    const U64 loss = static_cast<U64>(status.loss) + status.overrun;
    if (loss == streamLoss) {
        return;
    }
    streamLoss = loss;
    const U64 margin = 2 * PacketSamples;
    timestamps.lose(status.timestamp > std::numeric_limits<U64>::max() - margin
        ? std::numeric_limits<U64>::max()
        : status.timestamp + margin);
}

U64 LimeSdrStream::receive(std::span<std::byte* const> data, const U64 count,
                           lime::StreamMeta& meta, const std::chrono::milliseconds wait) {
    U64 delivered = 0;
    if (streamFormat == DataType::CI16) {
        std::array<lime::complex12_t*, MaxReceiveChannels> destinations{};
        for (U64 i = 0; i < data.size(); ++i) {
            destinations[i] = reinterpret_cast<lime::complex12_t*>(data[i]);
        }
        delivered = stream->StreamRx(destinations.data(), static_cast<uint32_t>(count), &meta,
                                     wait);
    } else {
        std::array<lime::complex32f_t*, MaxReceiveChannels> destinations{};
        for (U64 i = 0; i < data.size(); ++i) {
            destinations[i] = reinterpret_cast<lime::complex32f_t*>(data[i]);
        }
        delivered = stream->StreamRx(destinations.data(), static_cast<uint32_t>(count), &meta,
                                     wait);
    }
    if (delivered > 0) {
        received = Clock::now();
    }
    return delivered;
}

Device::Capture LimeSdrStream::read(std::span<std::byte> samples,
                                    const std::chrono::milliseconds timeout) {
    std::array destinations{samples};
    return readChannels(destinations, timeout);
}

Device::Capture LimeSdrStream::readChannels(std::span<std::span<std::byte>> samples,
                                            const std::chrono::milliseconds timeout) {
    const auto finish = [&](Device::Capture capture) {
        capture.generation = queueGeneration;
        capture.frequency = appliedSettings.frequency;
        capture.sampleRate = appliedSettings.sampleRate;
        capture.epoch = timestamps.epoch();
        return capture;
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
    if (samples.size() != receiveChannels || samples.front().size() < sampleBytes() ||
        std::any_of(samples.begin(), samples.end(), [&](const auto channel) {
            return channel.size() != samples.front().size();
        })) {
        return {.status = Status::Failure, .error = "invalid receive channel buffers"};
    }
    const U64 size = samples.front().size() / sampleBytes();

    const auto deadline = Clock::now() + timeout;
    while (const U64 pending = timestamps.pending()) {
        const U64 count = std::min<U64>(pending, drains.front().size() / sampleBytes());
        lime::StreamMeta meta = {};
        const U64 delivered = receive(drainPointers, count, meta, Remaining(deadline));
        poll();
        if (delivered == 0) {
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
    lime::StreamMeta meta = {};
    const U64 count = receive(outputs, size, meta, Remaining(deadline));
    poll();
    if (count == 0) {
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
            std::memmove(channel.data(), channel.data() + decision.skip * sampleBytes(),
                         (count - decision.skip) * sampleBytes());
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
        if (device->SetGain(0, lime::TRXDir::Rx, static_cast<U8>(i), lime::eGainTypes::GENERIC,
                            target + gainOffset) != lime::OpStatus::Success) {
            JST_ERROR("[SDR_LIMESDR] Failed to set the RX{} gain to {:.0f} dB: {}", i + 1,
                      target, lime::GetLastErrorMessage());
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
