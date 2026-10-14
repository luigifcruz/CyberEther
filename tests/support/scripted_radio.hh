#ifndef JETSTREAM_TESTS_SUPPORT_SCRIPTED_RADIO_HH
#define JETSTREAM_TESTS_SUPPORT_SCRIPTED_RADIO_HH

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "domains/io/sdr/devices/device.hh"
#include "domains/io/sdr/tools/samples.hh"
#include "jetstream/logger.hh"

// Marker samples encode frequency in MHz and generation; tone scenes use the
// tuned LO and applied sample rate.

namespace TestRadio {

using namespace Jetstream;

struct Tone {
    F64 frequency = 0.0;
    F64 amplitude = 1.0;
};

struct Script {
    std::mutex mutex;
    std::vector<std::string> lifecycle;
    std::vector<Radio::Device::Info> devices;
    std::string failAt;
    std::atomic<long> budget{-1};
    std::atomic<U64> opens{0};
    std::optional<Radio::Device::Capture> nextRead;
    std::vector<Tone> tones;
    F64 rateOffset = 0.0;
    F64 frequencyOffset = 0.0;
    std::optional<Radio::Device::FirmwareSweep> firmware;
    std::vector<Radio::Device::Range> sampleRates;
    U64 readLimit = 0;
    std::chrono::milliseconds readDelay{0};
    std::atomic<U64> largestRead{0};
    std::atomic<unsigned> activeReads{0};
    std::atomic<U64> concurrentChanges{0};

    void reset() {
        std::lock_guard lock(mutex);
        lifecycle.clear();
        devices = {{.serial = "A", .label = "Scripted Radio"}};
        failAt.clear();
        budget = -1;
        opens = 0;
        nextRead.reset();
        tones.clear();
        rateOffset = 0.0;
        frequencyOffset = 0.0;
        firmware.reset();
        sampleRates.clear();
        readLimit = 0;
        readDelay = std::chrono::milliseconds(0);
        largestRead = 0;
        activeReads = 0;
        concurrentChanges = 0;
    }

    void record(const std::string& call) {
        if (activeReads > 0) {
            ++concurrentChanges;
        }
        std::lock_guard lock(mutex);
        lifecycle.push_back(call);
    }

    std::vector<std::string> calls() {
        std::lock_guard lock(mutex);
        return lifecycle;
    }

    bool fails(const std::string& call) {
        std::lock_guard lock(mutex);
        return failAt == call;
    }
};

inline Script& TestScript() {
    static Script script;
    return script;
}

class ScriptedDevice final : public Radio::Device {
 public:
    explicit ScriptedDevice(const Radio::Device::Info& info) {
        deviceInfo = info;
        deviceCapabilities.frequencyRanges = {{10.0e6, 2000.0e6}};
        deviceCapabilities.sampleRateRanges = {{1.0e6, 10.0e6}};
        deviceCapabilities.antennas = {"RX1", "RX2"};
        deviceCapabilities.automaticGain = true;
        deviceCapabilities.biasTee = true;
        deviceCapabilities.sweepSampleRate = 2.0e6;
        deviceCapabilities.usableBandwidth = 0.75;
        deviceCapabilities.dcExclusion = 0.0;
        appliedSettings.gainRange = Radio::Device::Range{0.0, 60.0};
        defaultSettlingTime = 1.0e-3;
        {
            std::lock_guard lock(TestScript().mutex);
            deviceCapabilities.firmwareSweep = TestScript().firmware;
            if (TestScript().firmware) {
                deviceCapabilities.sampleRateRanges = {{1.0e6, 20.0e6}};
            }
            if (!TestScript().sampleRates.empty()) {
                deviceCapabilities.sampleRateRanges = TestScript().sampleRates;
            }
        }
        ++TestScript().opens;
        TestScript().record("open");
    }

    ~ScriptedDevice() override {
        stop();
        if (biasTeeEnabled) {
            TestScript().record("biastee:false");
        }
        TestScript().record("close");
    }

    Result start(const Radio::Device::Settings& settings) override {
        const auto call = jst::fmt::format("start:{:.0f}", settings.frequency / 1e6);
        TestScript().record(call);
        if (TestScript().fails("start") || TestScript().fails(call)) {
            return Result::ERROR;
        }
        if (settings.biasTee != biasTeeEnabled) {
            TestScript().record(settings.biasTee ? "biastee:true" : "biastee:false");
            biasTeeEnabled = settings.biasTee;
        }
        requestedSettings = settings;
        appliedSettings.frequency = settings.frequency + TestScript().frequencyOffset;
        appliedSettings.sampleRate = settings.sampleRate + TestScript().rateOffset;
        appliedSettings.antenna = settings.antenna;
        appliedSettings.automaticGain = settings.automaticGain;
        appliedSettings.gain = resolveGain(settings.gain, appliedSettings.gainRange);
        appliedSettings.biasTee = settings.biasTee;
        guardLength = guardSamples(settings, appliedSettings.sampleRate);
        if (guardLength == std::numeric_limits<U64>::max()) {
            return Result::ERROR;
        }
        nextGeneration();
        started = true;
        return Result::SUCCESS;
    }

    void stop() override {
        if (started) {
            TestScript().record("stop");
        }
        started = false;
    }

    // Frequency-only path, like the adapters: other settings stay applied.
    Result tune(const F64 frequency) override {
        const auto call = jst::fmt::format("tune:{:.0f}", frequency / 1e6);
        TestScript().record(call);
        nextGeneration();
        if (TestScript().fails(call)) {
            return Result::ERROR;
        }
        requestedSettings.frequency = frequency;
        appliedSettings.frequency = frequency + TestScript().frequencyOffset;
        return Result::SUCCESS;
    }

    Radio::Device::Capture read(std::span<CF32> samples, const std::chrono::milliseconds) override {
        ++TestScript().activeReads;
        struct Reading {
            ~Reading() { --TestScript().activeReads; }
        } reading;
        if (!started) {
            return {.status = Radio::Device::Status::Failure, .error = "stopped"};
        }
        {
            std::lock_guard lock(TestScript().mutex);
            if (auto scripted = TestScript().nextRead) {
                TestScript().nextRead.reset();
                scripted->generation = generation();
                return *scripted;
            }
        }
        auto& budget = TestScript().budget;
        U64 count = samples.size();
        U64 largest = TestScript().largestRead.load();
        while (count > largest && !TestScript().largestRead.compare_exchange_weak(largest, count)) {
        }
        std::chrono::milliseconds delay{0};
        {
            std::lock_guard lock(TestScript().mutex);
            if (TestScript().readLimit > 0) {
                count = std::min(count, TestScript().readLimit);
            }
            delay = TestScript().readDelay;
        }
        if (delay.count() > 0) {
            std::this_thread::sleep_for(delay);
        }
        if (const long allowed = budget.load(); allowed == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return {.status = Radio::Device::Status::Timeout, .generation = generation()};
        } else if (allowed > 0) {
            count = std::min<U64>(count, static_cast<U64>(allowed));
            budget -= static_cast<long>(count);
        }

        std::vector<Tone> tones;
        {
            std::lock_guard lock(TestScript().mutex);
            tones = TestScript().tones;
        }
        if (tones.empty()) {
            const CF32 value{static_cast<F32>(appliedSettings.frequency / 1e6),
                             static_cast<F32>(generation())};
            std::fill_n(samples.begin(), count, value);
        } else {
            // The scene runs on one continuous clock across retunes.
            const F64 rate = appliedSettings.sampleRate;
            for (U64 i = 0; i < count; ++i) {
                const F64 time = static_cast<F64>(clock + i) / rate;
                std::complex<F64> sample;
                for (const auto& tone : tones) {
                    // Models the front-end anti-alias filter.
                    if (std::abs(tone.frequency - appliedSettings.frequency) >= rate / 2.0) {
                        continue;
                    }
                    const F64 phase = 2.0 * std::numbers::pi * (tone.frequency - appliedSettings.frequency) * time;
                    sample += std::polar(tone.amplitude, phase);
                }
                samples[i] = CF32{static_cast<F32>(sample.real()), static_cast<F32>(sample.imag())};
            }
        }
        clock += count;
        return {
            .status = Radio::Device::Status::Samples,
            .count = count,
            .generation = generation(),
            .frequency = appliedSettings.frequency,
            .sampleRate = appliedSettings.sampleRate,
        };
    }

    Result setAutomaticGain(const bool automatic) override {
        const auto call = automatic ? "mode:true" : "mode:false";
        TestScript().record(call);
        if (TestScript().fails(call)) {
            return Result::ERROR;
        }
        requestedSettings.automaticGain = automatic;
        appliedSettings.automaticGain = automatic;
        return Result::SUCCESS;
    }

    Result setGain(const F64 gain) override {
        const auto call = jst::fmt::format("gain:{:.0f}", gain);
        TestScript().record(call);
        if (TestScript().fails(call)) {
            return Result::ERROR;
        }
        requestedSettings.gain = gain;
        appliedSettings.gain = resolveGain(gain, appliedSettings.gainRange);
        return Result::SUCCESS;
    }

    Result prepareSweep(const SweepRequest& request) override {
        const auto call = jst::fmt::format("prepare:{}", request.dwell);
        TestScript().record(call);
        if (TestScript().fails(call)) {
            return Result::ERROR;
        }
        return Device::prepareSweep(request);
    }

 private:
    bool started = false;
    bool biasTeeEnabled = false;
    U64 clock = 0;
};

inline Radio::Device::Driver ScriptedDriver() {
    return {
        .name = "scripted",
        .enumerate = [] {
            std::lock_guard lock(TestScript().mutex);
            return TestScript().devices;
        },
        .create = [](const Radio::Device::Info& info, std::unique_ptr<Radio::Device>& device) {
            if (TestScript().fails("open")) {
                return Result::ERROR;
            }
            device = std::make_unique<ScriptedDevice>(info);
            return Result::SUCCESS;
        },
    };
}

struct ScriptedDrivers {
    ScriptedDrivers() {
        TestScript().reset();
        Radio::Device::Override(std::vector<Radio::Device::Driver>{ScriptedDriver()});
    }

    ~ScriptedDrivers() {
        Radio::Device::Override(std::nullopt);
        TestScript().reset();
    }
};

constexpr const char* Selector = "driver=scripted,serial=A";

template<typename Predicate>
bool WaitFor(Predicate predicate, const std::chrono::milliseconds limit = std::chrono::seconds(2)) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace TestRadio

#endif  // JETSTREAM_TESTS_SUPPORT_SCRIPTED_RADIO_HH
