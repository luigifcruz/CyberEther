#include "base.hh"
#include "tools/settings.hh"

#include "../stream.hh"
#include "../../tools/capture_queue.hh"
#include "../../tools/samples.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>

#include <birriasdr.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

using BirriaSdrSettings::InFlightTransfers;
using BirriaSdrSettings::StageGains;
using BirriaSdrSettings::StaleSamples;
using BirriaSdrSettings::TransferSamples;

constexpr F64 SettlingTime = 1.0e-3;
constexpr U32 MaxRestarts = 64;
constexpr F32 SampleScale = 8192.0f;
constexpr auto ControlTimeout = std::chrono::seconds(5);
constexpr auto ControlRetry = std::chrono::milliseconds(10);

template<typename Call>
int Control(const Call& call) {
    const auto deadline = std::chrono::steady_clock::now() + ControlTimeout;
    int status = call();
    while (status == BIRRIASDR_ERROR_BUSY && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(ControlRetry);
        status = call();
    }
    return status;
}

class BirriaSdrStream final : public Stream {
 public:
    ~BirriaSdrStream() override;

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
    static void Callback(const int16_t* iq, uint32_t count, uint64_t firstSample, void* context);
    static void Recovery(const birriasdr_recovery_event* event, void* context);

    Result setFrequency(F64 frequency);
    Result applyGainMode(bool automatic, F64 gain);
    Result applyGain(F64 gain, bool force);

    birriasdr_dev_t* handle = nullptr;
    CaptureQueue queue;
    std::thread worker;
    std::atomic<bool> stopping{false};
    std::atomic<U64> queueGeneration{0};
    std::mutex finishedMutex;
    std::condition_variable finishedCondition;
    bool finished = true;
};

BirriaSdrStream::~BirriaSdrStream() {
    close();
    if (handle) {
        birriasdr_close(handle);
    }
}

Result BirriaSdrStream::connect(const Device::Info& info) {
    std::optional<U32> index;
    const int count = birriasdr_get_device_count();
    for (int i = 0; i < count && !index; ++i) {
        birriasdr_device_info candidate = {};
        if (birriasdr_get_device_info(static_cast<U32>(i), &candidate) == BIRRIASDR_OK &&
            jst::fmt::format("{}-{}", candidate.bus, candidate.address) == info.serial) {
            index = static_cast<U32>(i);
        }
    }
    if (!index) {
        JST_ERROR("[SDR_BIRRIASDR] Device '{}' is no longer available.", info.serial);
        return Result::INCOMPLETE;
    }
    if (const int status = birriasdr_open(&handle, *index); status != BIRRIASDR_OK) {
        handle = nullptr;
        JST_ERROR("[SDR_BIRRIASDR] Failed to open device '{}': {}", info.label,
                  birriasdr_strerror(status));
        return Result::ERROR;
    }
    deviceInfo = info;

    JST_CHECK(BirriaSdrSettings::ReadCapabilities(
        [this](const U32 index, uint64_t& minimum, uint64_t& maximum) {
            return birriasdr_get_tuning_band(handle, index, &minimum, &maximum);
        }, birriasdr_get_sample_rate(handle), deviceCapabilities));
    defaultSettlingTime = SettlingTime;
    appliedSettings.gainRange = Device::Range{0.0, static_cast<F64>(BirriaSdrSettings::MaxGain)};
    return Result::SUCCESS;
}

Result BirriaSdrStream::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_BIRRIASDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_BIRRIASDR] Frequency ({:.3f} MHz) not supported. Use {}.",
                  settings.frequency / 1e6,
                  Device::Range::Describe(deviceCapabilities.frequencyRanges));
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_BIRRIASDR] Antenna selection is not supported.");
        return Result::ERROR;
    }
    if (settings.biasTee) {
        JST_WARN("[SDR_BIRRIASDR] Bias-T is not supported by BirriaSDR.");
    }

    const auto rate = static_cast<uint32_t>(std::lround(settings.sampleRate));
    if (const int status = birriasdr_set_sample_rate(handle, rate); status != BIRRIASDR_OK) {
        JST_ERROR("[SDR_BIRRIASDR] Failed to set the sample rate: {}",
                  birriasdr_strerror(status));
        return Result::ERROR;
    }
    JST_CHECK(setFrequency(settings.frequency));

    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    requestedSettings = settings;
    appliedSettings.sampleRate = birriasdr_get_sample_rate(handle);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = false;
    return Result::SUCCESS;
}

void BirriaSdrStream::Callback(const int16_t* iq, const uint32_t count, uint64_t, void* context) {
    auto* stream = static_cast<BirriaSdrStream*>(context);
    if (stream->stopping) {
        return;
    }
    stream->queue.push(stream->queueGeneration, count,
                       [iq](CF32* output, const U64 offset, const U64 length) {
                           ConvertI16(iq + offset * 2, output, length, SampleScale);
                       });
}

void BirriaSdrStream::Recovery(const birriasdr_recovery_event* event, void* context) {
    auto* stream = static_cast<BirriaSdrStream*>(context);
    if (stream->stopping) {
        return;
    }
    JST_WARN("[SDR_BIRRIASDR] Capture discontinuity ({}).",
             birriasdr_strerror(event->error));
    stream->queue.push(stream->queueGeneration, 0, [](CF32*, U64, U64) {}, 1);
}

Result BirriaSdrStream::open() {
    if (const int status = birriasdr_set_recovery(handle, MaxRestarts,
                                                  &BirriaSdrStream::Recovery, this);
        status != BIRRIASDR_OK) {
        JST_ERROR("[SDR_BIRRIASDR] Failed to enable stream recovery: {}",
                  birriasdr_strerror(status));
        return Result::ERROR;
    }

    const F64 sampleRate = appliedSettings.sampleRate;
    const U64 capacity = std::max<U64>(static_cast<U64>(sampleRate / 4.0),
                                       static_cast<U64>(TransferSamples) * InFlightTransfers);
    JST_CHECK(queue.resize(capacity));
    queueGeneration = nextGeneration();
    queue.reset(queueGeneration, guard(), appliedSettings.frequency, sampleRate);

    finished = false;
    try {
        worker = std::thread([this] {
            const int status = birriasdr_read_async(handle, &BirriaSdrStream::Callback, this,
                                                    InFlightTransfers, TransferSamples);
            if (status != BIRRIASDR_OK && !stopping) {
                queue.fail(status == BIRRIASDR_ERROR_NO_DEVICE ? Device::Status::Disconnected
                                                             : Device::Status::Failure,
                           jst::fmt::format("reception stopped ({})",
                                            birriasdr_strerror(status)));
            }
            {
                std::lock_guard lock(finishedMutex);
                finished = true;
            }
            finishedCondition.notify_all();
        });
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_BIRRIASDR] Failed to start the receive thread: {}", e.what());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void BirriaSdrStream::close() {
    if (!worker.joinable()) {
        return;
    }
    stopping = true;
    std::unique_lock lock(finishedMutex);
    while (!finished) {
        birriasdr_cancel_async(handle);
        finishedCondition.wait_for(lock, std::chrono::milliseconds(10));
    }
    lock.unlock();
    worker.join();
    stopping = false;
    queue.close();
}

Result BirriaSdrStream::setFrequency(const F64 frequency) {
    const auto target = static_cast<uint64_t>(std::llround(frequency));
    if (const int status = Control([&] { return birriasdr_set_center_freq(handle, target); });
        status != BIRRIASDR_OK) {
        JST_ERROR("[SDR_BIRRIASDR] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                  birriasdr_strerror(status));
        return Result::ERROR;
    }
    appliedSettings.frequency = static_cast<F64>(birriasdr_get_center_freq(handle));
    return Result::SUCCESS;
}

Result BirriaSdrStream::tune(const F64 frequency) {
    JST_CHECK(setFrequency(frequency));
    requestedSettings.frequency = frequency;
    if (!worker.joinable()) {
        return Result::SUCCESS;
    }
    queueGeneration = nextGeneration();
    queue.retune(queueGeneration, 0, guard() + StaleSamples(appliedSettings.sampleRate),
                 appliedSettings.frequency);
    return Result::SUCCESS;
}

Device::Capture BirriaSdrStream::read(std::span<CF32> samples,
                                      const std::chrono::milliseconds timeout) {
    return queue.pop(samples, timeout);
}

Result BirriaSdrStream::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result BirriaSdrStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 BirriaSdrStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range));
}

Result BirriaSdrStream::applyGainMode(const bool automatic, const F64 gain) {
    if (const int status = Control([&] {
            return birriasdr_set_tuner_gain_mode(handle, automatic ? 0 : 1);
        });
        status != BIRRIASDR_OK) {
        JST_ERROR("[SDR_BIRRIASDR] Failed to set the gain mode: {}", birriasdr_strerror(status));
        return Result::ERROR;
    }
    appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result BirriaSdrStream::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (appliedSettings.automaticGain || (!force && target == appliedSettings.gain)) {
        appliedSettings.gain = target;
        return Result::SUCCESS;
    }
    const auto stages = StageGains(static_cast<U64>(target));
    if (const int status = Control([&] { return birriasdr_set_gain_codes(handle, &stages); });
        status != BIRRIASDR_OK) {
        JST_ERROR("[SDR_BIRRIASDR] Failed to set the gain to {:.0f}: {}", target,
                  birriasdr_strerror(status));
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result BirriaSdr::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<BirriaSdrStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
