#include "base.hh"
#include "tools/settings.hh"

#include "../sweep.hh"
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

class BirriaSdrSweep final : public Sweep::Retune {
 public:
    ~BirriaSdrSweep() override;

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

BirriaSdrSweep::~BirriaSdrSweep() {
    close();
    if (handle) {
        birriasdr_close(handle);
    }
}

Result BirriaSdrSweep::connect(const Device::Info& info) {
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

Result BirriaSdrSweep::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_BIRRIASDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
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

    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    requestedSettings = settings;
    appliedSettings.sampleRate = birriasdr_get_sample_rate(handle);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = false;
    return Result::SUCCESS;
}

void BirriaSdrSweep::Callback(const int16_t* iq, const uint32_t count, uint64_t, void* context) {
    auto* sweep = static_cast<BirriaSdrSweep*>(context);
    if (sweep->stopping) {
        return;
    }
    sweep->queue.push(sweep->queueGeneration, count,
                      [iq](CF32* output, const U64 offset, const U64 length) {
                          ConvertI16(iq + offset * 2, output, length, SampleScale);
                      });
}

void BirriaSdrSweep::Recovery(const birriasdr_recovery_event* event, void* context) {
    auto* sweep = static_cast<BirriaSdrSweep*>(context);
    if (sweep->stopping) {
        return;
    }
    JST_WARN("[SDR_BIRRIASDR] Capture discontinuity ({}).",
             birriasdr_strerror(event->error));
    sweep->queue.push(sweep->queueGeneration, 0, [](CF32*, U64, U64) {}, 1);
}

Result BirriaSdrSweep::open() {
    JST_CHECK(setFrequency(sweepRequest.frequencies.front()));
    if (const int status = birriasdr_set_recovery(handle, MaxRestarts,
                                                  &BirriaSdrSweep::Recovery, this);
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
    queue.reset(queueGeneration, guard(), tunedFrequency, sampleRate);

    finished = false;
    try {
        worker = std::thread([this] {
            const int status = birriasdr_read_async(handle, &BirriaSdrSweep::Callback, this,
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

void BirriaSdrSweep::close() {
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

Result BirriaSdrSweep::setFrequency(const F64 frequency) {
    const auto target = static_cast<uint64_t>(std::llround(frequency));
    if (const int status = Control([&] { return birriasdr_set_center_freq(handle, target); });
        status != BIRRIASDR_OK) {
        JST_ERROR("[SDR_BIRRIASDR] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                  birriasdr_strerror(status));
        return Result::ERROR;
    }
    tunedFrequency = static_cast<F64>(birriasdr_get_center_freq(handle));
    return Result::SUCCESS;
}

Result BirriaSdrSweep::tune(const U64 hop) {
    if (!worker.joinable()) {
        JST_ERROR("[SDR_BIRRIASDR] Cannot retune a stopped sweep.");
        return Result::ERROR;
    }
    JST_CHECK(setFrequency(sweepRequest.frequencies[hop]));
    queueGeneration = nextGeneration();
    queue.retune(queueGeneration, 0, guard() + StaleSamples(appliedSettings.sampleRate),
                 tunedFrequency);
    return Result::SUCCESS;
}

Device::Capture BirriaSdrSweep::read(std::span<CF32> samples,
                                     const std::chrono::milliseconds timeout) {
    return queue.pop(samples, timeout);
}

Result BirriaSdrSweep::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result BirriaSdrSweep::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 BirriaSdrSweep::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Sweep::resolveGain(gain, range));
}

Result BirriaSdrSweep::applyGainMode(const bool automatic, const F64 gain) {
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

Result BirriaSdrSweep::applyGain(const F64 gain, const bool force) {
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

Result BirriaSdr::Sweep(const Device::Info& info, std::unique_ptr<Radio::Sweep>& sweep) {
    auto created = std::make_unique<BirriaSdrSweep>();
    JST_CHECK(created->connect(info));
    sweep = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
