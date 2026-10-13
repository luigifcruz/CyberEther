#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "domains/io/sdr/devices/device.hh"
#include "domains/io/sdr/tools/capture_queue.hh"
#include "domains/io/sdr/tools/stream_access.hh"
#include "domains/io/sdr/tools/samples.hh"
#include "domains/io/sdr/tools/stream_buffer.hh"
#include "domains/io/sdr/tools/timestamp_filter.hh"
#include "flowgraph_fixture.hh"
#include "jetstream/domains/io/sdr/block.hh"
#include "jetstream/domains/io/sdr/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"
#include "module_impl.hh"
#include "scripted_radio.hh"

using namespace Jetstream;

namespace {

using namespace TestRadio;

struct ScriptedFlowgraph : ScriptedDrivers, FlowgraphFixture {};

// Vendor calls are injected, but tuning transactions, epoch validity, and
// captureHop use the same shared paths as the restart-based adapters.
class RestartDevice final : public Radio::Device {
 public:
    using Radio::Device::captureHop;

    bool setterFails = false;
    bool restartFails = false;
    bool restoreFails = false;
    bool inactive = false;
    bool streaming = false;
    U64 starts = 0;
    U64 reads = 0;
    U64 guardReads = 0;

    Result start(const Radio::Device::Settings& settings) override {
        ++starts;
        if (restoreFails) {
            return Result::ERROR;
        }
        requestedSettings = settings;
        appliedSettings.frequency = settings.frequency;
        appliedSettings.sampleRate = settings.sampleRate;
        guardLength = guardSamples(settings, settings.sampleRate);
        return startStream();
    }

    void stop() override {
        streaming = false;
    }

    Result tune(const F64 frequency) override {
        return restart([&] {
            if (setterFails) {
                return Result::ERROR;
            }
            requestedSettings.frequency = frequency;
            appliedSettings.frequency = frequency;
            if (restartFails) {
                return Result::ERROR;
            }
            return startStream();
        });
    }

    Radio::Device::Capture read(std::span<CF32> samples, const std::chrono::milliseconds timeout) override {
        ++reads;
        if (!streaming) {
            return {.status = Radio::Device::Status::Failure};
        }
        if (inactive) {
            std::this_thread::sleep_for(timeout);
            return {.status = Radio::Device::Status::Timeout};
        }
        const U64 count = samples.size();
        const auto decision = timestamps.accept(timestamp, count);
        timestamp += count;
        if (decision.status == Radio::Device::Status::Timeout) {
            ++guardReads;
        }
        const U64 first = timestamp - count + decision.skip;
        for (U64 i = 0; i < samples.size(); ++i) {
            samples[i] = CF32{static_cast<F32>(first + i), 0.0f};
        }
        return {.status = decision.status, .count = count - decision.skip,
                .generation = generation(), .timestamp = first,
                .epoch = timestamps.epoch(), .error = decision.error};
    }

    F64 requestedFrequency() const {
        return requestedSettings.frequency;
    }

 private:
    Result startStream() {
        timestamp = 0;
        timestamps.restart(guardLength, 0);
        nextGeneration();
        streaming = true;
        return Result::SUCCESS;
    }

    Radio::TimestampFilter timestamps;
    U64 timestamp = 0;
};

// Delivers a running sample counter, so a capture shows exactly which samples
// it kept. Reads listed in `lossAt` report a gap of `Lost` samples instead,
// reads in `staleAt` or all reads with `staleReads` deliver samples under the
// previous generation, and `emptyReads` returns samples without any data.
class CounterDevice final : public Radio::Device {
 public:
    using Radio::Device::captureHop;

    static constexpr U64 Lost = 7;

    U64 starts = 0;
    U64 stops = 0;
    U64 reads = 0;
    U64 readLimit = 40;
    bool emptyReads = false;
    bool staleReads = false;
    std::vector<U64> lossAt;
    std::vector<U64> staleAt;
    std::vector<F64> tunes;

    Result start(const Radio::Device::Settings& settings) override {
        ++starts;
        requestedSettings = settings;
        appliedSettings.frequency = settings.frequency;
        appliedSettings.sampleRate = settings.sampleRate;
        nextGeneration();
        return Result::SUCCESS;
    }

    void stop() override {
        ++stops;
    }

    Result tune(const F64 frequency) override {
        tunes.push_back(frequency);
        requestedSettings.frequency = frequency;
        appliedSettings.frequency = frequency;
        nextGeneration();
        return Result::SUCCESS;
    }

    Radio::Device::Capture read(std::span<CF32> samples, const std::chrono::milliseconds) override {
        const U64 index = reads++;
        if (std::find(lossAt.begin(), lossAt.end(), index) != lossAt.end()) {
            counter += Lost;
            return {.status = Radio::Device::Status::Discontinuity, .generation = generation(),
                    .lost = Lost};
        }
        if (emptyReads || staleReads) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (emptyReads) {
            return {.status = Radio::Device::Status::Samples, .generation = generation(),
                    .frequency = appliedSettings.frequency};
        }
        const bool stale = staleReads ||
                           std::find(staleAt.begin(), staleAt.end(), index) != staleAt.end();
        const U64 count = std::min<U64>(samples.size(), readLimit);
        for (U64 i = 0; i < count; ++i) {
            samples[i] = CF32{static_cast<F32>(counter++), 0.0f};
        }
        return {.status = Radio::Device::Status::Samples, .count = count,
                .generation = generation() - (stale ? 1 : 0),
                .frequency = appliedSettings.frequency};
    }

 private:
    U64 counter = 0;
};

// Models a callback adapter: a producer thread fills USB-sized transfers from
// the live tuner and delivers each one after it completes, so a transfer can
// straddle a retune or arrive after it. Hops retune exactly like the
// RTL-SDR, Airspy, and HackRF adapters.
class StreamingDevice final : public Radio::Device {
 public:
    static constexpr U64 TransferSamples = 256;

    std::atomic<U64> starts{0};
    std::atomic<U64> stops{0};
    std::atomic<U64> transfers{0};

    ~StreamingDevice() override {
        stop();
    }

    Result start(const Radio::Device::Settings& settings) override {
        stop();
        ++starts;
        requestedSettings = settings;
        appliedSettings.frequency = settings.frequency;
        appliedSettings.sampleRate = settings.sampleRate;
        guardLength = guardSamples(settings, settings.sampleRate);
        tuner = settings.frequency;
        JST_CHECK(queue.resize(4 * TransferSamples));
        streamGeneration = nextGeneration();
        queue.reset(streamGeneration, guardLength, appliedSettings.frequency,
                    appliedSettings.sampleRate);
        streaming = true;
        producer = std::thread([this] {
            std::vector<CF32> transfer(TransferSamples);
            while (streaming) {
                for (U64 i = 0; i < TransferSamples; ++i) {
                    transfer[i] = CF32{static_cast<F32>(tuner.load() / 1e6), 0.0f};
                    if (i == TransferSamples / 2) {
                        std::this_thread::yield();
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
                queue.push(streamGeneration, TransferSamples,
                           [&](CF32* output, const U64 offset, const U64 count) {
                               std::copy_n(transfer.data() + offset, count, output);
                           });
                ++transfers;
            }
        });
        return Result::SUCCESS;
    }

    void stop() override {
        if (!producer.joinable()) {
            return;
        }
        ++stops;
        streaming = false;
        producer.join();
        queue.close();
    }

    Radio::Device::Capture read(std::span<CF32> samples, const std::chrono::milliseconds timeout) override {
        return queue.pop(samples, timeout);
    }

 protected:
    Result tuneHop(const U64 hop) override {
        tuner = sweepRequest.frequencies[hop];
        requestedSettings.frequency = tuner;
        appliedSettings.frequency = tuner;
        streamGeneration = nextGeneration();
        queue.retune(streamGeneration, 1, guardLength, appliedSettings.frequency);
        return Result::SUCCESS;
    }

 private:
    Radio::CaptureQueue queue;
    std::thread producer;
    std::atomic<bool> streaming{false};
    std::atomic<F64> tuner{0.0};
    std::atomic<U64> streamGeneration{0};
};

struct SdrImplAccess : Modules::SdrImpl {
    static auto streamMember() {
        return &SdrImplAccess::stream;
    }
};

std::shared_ptr<Module> BuildSdrModule() {
    const auto implementations = Registry::ListAvailableModules("sdr");
    REQUIRE_FALSE(implementations.empty());
    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("sdr", implementation.device, implementation.runtime,
                                  implementation.provider, module) == Result::SUCCESS);
    return module;
}

struct ModuleCleanup {
    std::shared_ptr<Module> module;

    ~ModuleCleanup() {
        if (module->state() == Module::State::CREATED) {
            static_cast<void>(module->destroy());
        }
    }
};

Modules::Sdr ScriptedConfig() {
    Modules::Sdr config;
    config.deviceString = Selector;
    config.frequency = 100.0e6f;
    config.sampleRate = 2.0e6f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    config.bufferMultiplier = 4;
    return config;
}

F32 AttributeOf(const Tensor& tensor, const std::string& key) {
    return std::any_cast<F32>(tensor.attribute(key));
}

std::shared_ptr<NativeCpuRuntimeContext> RuntimeOf(const std::shared_ptr<Module>& module) {
    auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(module->context()->runtime());
    REQUIRE(runtime != nullptr);
    return runtime;
}

}  // namespace

//
// Backend.
//

TEST_CASE("SDR selections round-trip and reject malformed identities",
          "[modules][sdr][backend][selection]") {
    const Radio::Device::Info info{.driver = "rtlsdr", .serial = "00000001", .label = "RTL"};
    REQUIRE(info.selector() == "driver=rtlsdr,serial=00000001");
    const auto parsed = Radio::Device::Selection::Parse(info.selector());
    REQUIRE(parsed);
    REQUIRE(parsed->matches(info));
    REQUIRE_FALSE(parsed->matches({.driver = "rtlsdr", .serial = "00000002"}));
    REQUIRE_FALSE(parsed->matches({.driver = "airspy", .serial = "00000001"}));

    const auto empty = Radio::Device::Selection::Parse("driver=hackrf,serial=");
    REQUIRE(empty);
    REQUIRE(empty->serial.empty());

    for (const auto* invalid : {"", "driver=rtlsdr", "serial=1", "driver=,serial=1",
                                "driver=rtlsdr,serial=1,serial=2", "driver=rtlsdr,index=0",
                                "driver=rtlsdr;serial=1", "rtlsdr"}) {
        INFO(invalid);
        REQUIRE_FALSE(Radio::Device::Selection::Parse(invalid));
    }
}

TEST_CASE("SDR device labels keep distinct devices with duplicate names",
          "[modules][sdr][backend][selection]") {
    const auto labeled = Radio::Device::Label({
        {.driver = "rtlsdr", .serial = "A", .label = "RTL"},
        {.driver = "rtlsdr", .serial = "B", .label = "RTL"},
        {.driver = "rtlsdr", .serial = "B", .label = "RTL"},
        {.driver = "hackrf", .serial = "C", .label = ""},
    });
    REQUIRE(labeled.size() == 4);
    REQUIRE(labeled[0].first == "RTL [A]");
    REQUIRE(labeled[1].first == "RTL [B]");
    REQUIRE(labeled[2].first == "RTL [B] #2");
    REQUIRE(labeled[3].first == "hackrf");
    REQUIRE(labeled[2].second.serial == "B");
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR opening selects exactly one connected device",
                 "[modules][sdr][backend][selection]") {
    std::unique_ptr<Radio::Device> device;

    REQUIRE(Radio::Device::Open("not a selector", device) == Result::ERROR);
    REQUIRE(Radio::Device::Open("driver=missing,serial=A", device) == Result::INCOMPLETE);
    REQUIRE(Radio::Device::Open("driver=scripted,serial=B", device) == Result::INCOMPLETE);
    REQUIRE(TestScript().opens == 0);

    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().devices = {{.serial = "A", .label = "One"}, {.serial = "A", .label = "Two"}};
    }
    REQUIRE(Radio::Device::Open(Selector, device) == Result::INCOMPLETE);
    REQUIRE(TestScript().opens == 0);

    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().devices = {{.serial = "B", .label = "Other"}, {.serial = "A", .label = "One"}};
    }
    REQUIRE(Radio::Device::Open(Selector, device) == Result::SUCCESS);
    REQUIRE(device->info().serial == "A");
    REQUIRE(device->info().driver == "scripted");
    REQUIRE(TestScript().opens == 1);

    device.reset();
    TestScript().failAt = "open";
    REQUIRE(Radio::Device::Open(Selector, device) == Result::ERROR);
    REQUIRE(device == nullptr);
}

TEST_CASE("SDR capture queues reject samples from previous generations",
          "[modules][sdr][backend][capture]") {
    Radio::CaptureQueue queue;
    REQUIRE(queue.resize(64) == Result::SUCCESS);
    std::vector<CF32> output(64);
    const auto fill = [](const F32 value) {
        return [value](CF32* data, U64, const U64 count) {
            std::fill_n(data, count, CF32{value, 0.0f});
        };
    };

    queue.reset(1, 0, 100.0e6, 2.0e6);
    queue.push(1, 4, fill(1.0f));

    // The tune completed: several pre-tune transfers still arrive afterwards.
    queue.reset(2, 0, 200.0e6, 2.0e6);
    for (int i = 0; i < 3; ++i) {
        queue.push(1, 8, fill(1.0f));
    }
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Timeout);

    queue.push(2, 4, fill(2.0f));
    const auto capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.status == Radio::Device::Status::Samples);
    REQUIRE(capture.count == 4);
    REQUIRE(capture.generation == 2);
    REQUIRE(capture.frequency == 200.0e6);
    REQUIRE(output[0] == CF32{2.0f, 0.0f});
}

TEST_CASE("SDR capture queues count the settling guard from the boundary",
          "[modules][sdr][backend][capture]") {
    Radio::CaptureQueue queue;
    REQUIRE(queue.resize(64) == Result::SUCCESS);
    std::vector<CF32> output(64);
    queue.reset(7, 10, 0.0, 1.0);
    U64 index = 0;
    const auto ramp = [&index](CF32* data, const U64 offset, const U64 count) {
        for (U64 i = 0; i < count; ++i) {
            data[i] = CF32{static_cast<F32>(offset + i + index), 0.0f};
        }
    };
    queue.push(7, 6, ramp);
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Timeout);
    index = 6;
    queue.push(7, 6, ramp);
    const auto capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.count == 2);
    REQUIRE(output[0] == CF32{10.0f, 0.0f});
    REQUIRE(output[1] == CF32{11.0f, 0.0f});
}

TEST_CASE("SDR capture queues report overflow and device loss in stream order",
          "[modules][sdr][backend][capture]") {
    Radio::CaptureQueue queue;
    REQUIRE(queue.resize(8) == Result::SUCCESS);
    std::vector<CF32> output(8);
    const auto fill = [](const F32 value) {
        return [value](CF32* data, U64, const U64 count) {
            std::fill_n(data, count, CF32{value, 0.0f});
        };
    };
    queue.reset(1, 0, 0.0, 1.0);
    queue.push(1, 6, fill(1.0f));
    queue.push(1, 6, fill(2.0f));
    queue.push(1, 2, fill(3.0f), 5);

    auto capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.status == Radio::Device::Status::Samples);
    REQUIRE(capture.count == 6);
    REQUIRE(output[0] == CF32{1.0f, 0.0f});

    capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.status == Radio::Device::Status::Discontinuity);
    REQUIRE(capture.lost == 11);

    capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.status == Radio::Device::Status::Samples);
    REQUIRE(capture.count == 2);
    REQUIRE(output[0] == CF32{3.0f, 0.0f});

    queue.fail(Radio::Device::Status::Disconnected, "unplugged");
    capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.status == Radio::Device::Status::Disconnected);
    REQUIRE(capture.error == "unplugged");

    queue.close();
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Failure);
    queue.push(1, 2, fill(4.0f));
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Failure);
}

TEST_CASE("SDR sample conversions normalize vendor formats",
          "[modules][sdr][backend][conversion]") {
    const U8 unsignedIq[] = {0, 255, 127, 128};
    std::vector<CF32> output(2);
    Radio::ConvertU8(unsignedIq, output.data(), 2);
    REQUIRE(output[0] == CF32{-1.0f, 1.0f});
    REQUIRE_THAT(output[1].real(), Catch::Matchers::WithinAbs(-0.5f / 127.5f, 1e-7));
    REQUIRE_THAT(output[1].imag(), Catch::Matchers::WithinAbs(0.5f / 127.5f, 1e-7));

    const I8 signedIq[] = {-128, 127, 0, 64};
    Radio::ConvertI8(signedIq, output.data(), 2);
    REQUIRE(output[0] == CF32{-1.0f, 127.0f / 128.0f});
    REQUIRE(output[1] == CF32{0.0f, 0.5f});

    const I16 q11[] = {-2048, 2047, 1024, -1};
    Radio::ConvertI16(q11, output.data(), 2, 2048.0f);
    REQUIRE(output[0] == CF32{-1.0f, 2047.0f / 2048.0f});
    REQUIRE(output[1] == CF32{0.5f, -1.0f / 2048.0f});
}

TEST_CASE("SDR settling guards round up at the actual capture rate",
          "[modules][sdr][backend][timing]") {
    REQUIRE(Radio::GuardSamples(5.0e-3, 2.0e6) == 10000);
    REQUIRE(Radio::GuardSamples(1.0e-3, 2.4e6) == 2400);
    REQUIRE(Radio::GuardSamples(1.0e-6, 2.5e6) == 3);
    REQUIRE(Radio::GuardSamples(0.0, 2.0e6) == 0);
    const U64 invalid = std::numeric_limits<U64>::max();
    REQUIRE(Radio::GuardSamples(-1.0, 2.0e6) == invalid);
    REQUIRE(Radio::GuardSamples(std::numeric_limits<F64>::quiet_NaN(), 2.0e6) == invalid);
    REQUIRE(Radio::GuardSamples(std::numeric_limits<F32>::max() / 1e3, 2.0e6) == invalid);
    REQUIRE(Radio::GuardSamples(std::numeric_limits<F64>::max(), 2.0e6) == invalid);
    REQUIRE(Radio::GuardSamples(1.0, 0.0) == invalid);
    REQUIRE(Radio::GuardSamples(0.0, std::numeric_limits<F64>::infinity()) == invalid);

    const F64 limit = static_cast<F64>(invalid);
    const F64 below = std::nextafter(limit, 0.0);
    REQUIRE(Radio::GuardSamples(limit, 1.0) == invalid);
    REQUIRE(Radio::GuardSamples(below, 1.0) == static_cast<U64>(below));
}

TEST_CASE("SDR restart tuning restores setter and receive-start failures",
          "[modules][sdr][backend][lifecycle]") {
    RestartDevice device;
    const Radio::Device::Settings previous{.frequency = 100.0e6, .sampleRate = 20.0e6,
                                    .settlingTime = 0.002};
    REQUIRE(device.start(previous) == Result::SUCCESS);
    SECTION("setter failure") {
        device.setterFails = true;
    }
    SECTION("FIFO reset or receive-start failure") {
        device.restartFails = true;
    }
    SECTION("restoration also fails") {
        device.restartFails = true;
        device.restoreFails = true;
    }
    SECTION("success does not reapply all settings") {
        REQUIRE(device.tune(200.0e6) == Result::SUCCESS);
        REQUIRE(device.starts == 1);
        REQUIRE(device.applied().frequency == 200.0e6);
        REQUIRE(device.streaming);
        return;
    }
    REQUIRE(device.tune(200.0e6) == Result::ERROR);
    REQUIRE(device.starts == 2);
    REQUIRE(device.requestedFrequency() == previous.frequency);
    REQUIRE(device.applied().frequency == previous.frequency);
    REQUIRE(device.applied().sampleRate == previous.sampleRate);
    REQUIRE(device.streaming == !device.restoreFails);
    if (device.streaming) {
        std::vector<CF32> samples(40001);
        REQUIRE(device.read(samples, std::chrono::milliseconds(100)).status == Radio::Device::Status::Samples);
    }
}

TEST_CASE("SDR generic hop capture budgets elapsed settling time instead of guard reads",
          "[modules][sdr][backend][capture][lime]") {
    RestartDevice device;
    F64 guard = 0.002;
    U64 length = 200;
    bool cancel = false;
    SECTION("100 kHz bins and a 2 ms guard") {}
    SECTION("10 kHz bins and a 5 ms override") {
        guard = 0.005;
        length = 2000;
    }
    SECTION("cancellation during guard discard") {
        cancel = true;
    }
    SECTION("real inactivity still times out") {
        device.inactive = true;
    }
    REQUIRE(device.start({.frequency = 100.0e6, .sampleRate = 20.0e6,
                           .settlingTime = guard}) == Result::SUCCESS);
    REQUIRE(device.prepareSweep({.frequencies = {101.0e6}, .sampleRate = 20.0e6,
                                  .captureLength = length}) == Result::SUCCESS);
    std::vector<CF32> samples(length);
    Radio::Device::SweepCapture capture;
    U64 segments = 0;
    const auto result = device.captureHop(0, samples, segments, capture, [&] {
        return !cancel || device.reads < 30;
    });
    if (cancel && !device.inactive) {
        REQUIRE(result == Result::YIELD);
        REQUIRE(device.reads == 30);
    } else if (device.inactive) {
        REQUIRE(result == Result::TIMEOUT);
    } else {
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(device.guardReads > 21);
        REQUIRE(capture.samples.front().real() == static_cast<F32>(Radio::GuardSamples(guard, 20.0e6)));
        REQUIRE(capture.samples.size() == length);
    }
}

TEST_CASE("SDR capture queues relabel a running stream without keeping pre-retune samples",
          "[modules][sdr][backend][capture][retune]") {
    Radio::CaptureQueue queue;
    REQUIRE(queue.resize(64) == Result::SUCCESS);
    std::vector<CF32> output(64);
    const auto fill = [](const F32 value) {
        return [value](CF32* data, U64, const U64 count) {
            std::fill_n(data, count, CF32{value, 0.0f});
        };
    };
    const auto ramp = [](CF32* data, const U64 offset, const U64 count) {
        for (U64 i = 0; i < count; ++i) {
            data[i] = CF32{static_cast<F32>(offset + i), 0.0f};
        }
    };

    queue.reset(1, 0, 100.0e6, 2.0e6);
    queue.push(1, 8, fill(1.0f));
    REQUIRE(queue.pop(std::span(output).first(3), std::chrono::milliseconds(1)).count == 3);

    // The partly consumed transfer and everything queued behind it are gone.
    queue.push(1, 8, fill(1.0f));
    queue.retune(2, 1, 4, 200.0e6);
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Timeout);

    // A callback that loaded the old generation is rejected, and the first
    // accepted transfer may straddle the change, so it is dropped whole.
    queue.push(1, 8, fill(1.0f));
    queue.push(2, 8, fill(1.0f));
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Timeout);

    // The settling guard is counted from the next transfer.
    queue.push(2, 6, ramp);
    auto capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.status == Radio::Device::Status::Samples);
    REQUIRE(capture.count == 2);
    REQUIRE(capture.generation == 2);
    REQUIRE(capture.frequency == 200.0e6);
    REQUIRE(capture.sampleRate == 2.0e6);
    REQUIRE(output[0] == CF32{4.0f, 0.0f});

    // Loss reported before the retune never reaches the new hop.
    queue.push(2, 70, fill(2.0f));
    queue.retune(3, 0, 0, 300.0e6);
    queue.push(3, 4, fill(3.0f));
    capture = queue.pop(output, std::chrono::milliseconds(1));
    REQUIRE(capture.status == Radio::Device::Status::Samples);
    REQUIRE(output[0] == CF32{3.0f, 0.0f});

    // A dead stream stays dead, and a stopped stream stays stopped.
    queue.fail(Radio::Device::Status::Disconnected, "unplugged");
    queue.retune(4, 0, 0, 400.0e6);
    queue.push(4, 4, fill(4.0f));
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Disconnected);
    queue.reset(5, 0, 500.0e6, 2.0e6);
    queue.close();
    queue.retune(6, 0, 0, 600.0e6);
    queue.push(6, 4, fill(6.0f));
    REQUIRE(queue.pop(output, std::chrono::milliseconds(1)).status == Radio::Device::Status::Failure);
}

TEST_CASE("SDR sweeps retune a running stream and never label earlier samples with a hop",
          "[modules][sdr][backend][capture][retune]") {
    StreamingDevice device;
    REQUIRE(device.start({.frequency = 100.0e6, .sampleRate = 2.0e6, .settlingTime = 0.0}) ==
            Result::SUCCESS);
    REQUIRE(device.prepareSweep({.frequencies = {100.0e6, 200.0e6, 300.0e6, 400.0e6},
                                 .sampleRate = 2.0e6, .captureLength = 64,
                                 .dwell = 2}) == Result::SUCCESS);

    U64 captures = 0;
    U64 mislabeled = 0;
    std::vector<U64> hops;
    REQUIRE(device.runSweep([] { return true; }, [&](const Radio::Device::SweepCapture& capture) {
        for (const auto& sample : capture.samples) {
            if (sample.real() != static_cast<F32>(capture.frequency / 1e6)) {
                ++mislabeled;
            }
        }
        hops.push_back(capture.hop);
        return ++captures < 200;
    }) == Result::SUCCESS);

    REQUIRE(mislabeled == 0);
    REQUIRE(device.starts == 1);
    REQUIRE(device.stops == 0);
    REQUIRE(hops.size() == 200);
    for (U64 i = 0; i < hops.size(); ++i) {
        REQUIRE(hops[i] == i % 4);
    }
    REQUIRE(device.statistics().failedHops == 0);
}

TEST_CASE("SDR hop captures keep finished dwell segments across local loss",
          "[modules][sdr][backend][capture][dwell]") {
    CounterDevice device;
    REQUIRE(device.start({.frequency = 100.0e6, .sampleRate = 2.0e6}) == Result::SUCCESS);
    REQUIRE(device.prepareSweep({.frequencies = {100.0e6, 101.0e6}, .sampleRate = 2.0e6,
                                 .captureLength = 100, .dwell = 3}) == Result::SUCCESS);
    U64 failedHops = 0;
    U64 tunes = 1;
    U64 resumeAt = 120 + CounterDevice::Lost;
    SECTION("a gap inside the second segment only repeats that segment") {
        device.lossAt = {3};
    }
    SECTION("a failed hop is retuned in place and keeps the first segment") {
        for (U64 read = 3; read < 12; ++read) {
            device.lossAt.push_back(read);
        }
        failedHops = 1;
        tunes = 2;
        resumeAt = 120 + 9 * CounterDevice::Lost;
    }

    std::vector<F32> kept;
    REQUIRE(device.runSweep([] { return true; }, [&](const Radio::Device::SweepCapture& capture) {
        REQUIRE(capture.hop == 0);
        for (const auto& sample : capture.samples) {
            kept.push_back(sample.real());
        }
        return false;
    }) == Result::SUCCESS);

    REQUIRE(kept.size() == 300);
    for (U64 i = 0; i < 100; ++i) {
        REQUIRE(kept[i] == static_cast<F32>(i));
    }
    for (U64 i = 100; i < 300; ++i) {
        REQUIRE(kept[i] == static_cast<F32>(resumeAt + i - 100));
    }
    REQUIRE(device.statistics().failedHops == failedHops);
    REQUIRE(device.tunes == std::vector<F64>(tunes, 100.0e6));
    REQUIRE(device.starts == 1);
    REQUIRE(device.stops == 0);
}

TEST_CASE("SDR hop captures never bridge an FFT across samples of another generation",
          "[modules][sdr][backend][capture][dwell]") {
    CounterDevice device;
    REQUIRE(device.start({.frequency = 100.0e6, .sampleRate = 2.0e6}) == Result::SUCCESS);
    REQUIRE(device.prepareSweep({.frequencies = {100.0e6}, .sampleRate = 2.0e6,
                                 .captureLength = 100, .dwell = 3}) == Result::SUCCESS);
    // The stale read arrives 20 samples into the second FFT.
    device.staleAt = {3};

    std::vector<CF32> samples(300);
    U64 segments = 0;
    Radio::Device::SweepCapture capture;
    REQUIRE(device.captureHop(0, samples, segments, capture, [] { return true; }) ==
            Result::SUCCESS);
    for (U64 i = 0; i < 100; ++i) {
        REQUIRE(samples[i].real() == static_cast<F32>(i));
    }
    for (U64 i = 100; i < 300; ++i) {
        REQUIRE(samples[i].real() == static_cast<F32>(160 + i - 100));
    }
}

TEST_CASE("SDR hop captures stay bounded and cancellable without usable samples",
          "[modules][sdr][backend][capture][timing]") {
    CounterDevice device;
    REQUIRE(device.start({.frequency = 100.0e6, .sampleRate = 2.0e6}) == Result::SUCCESS);
    REQUIRE(device.prepareSweep({.frequencies = {100.0e6}, .sampleRate = 2.0e6,
                                 .captureLength = 100, .dwell = 3}) == Result::SUCCESS);
    SECTION("zero-count reads") {
        device.emptyReads = true;
    }
    SECTION("reads from another generation") {
        device.staleReads = true;
    }

    std::vector<CF32> samples(300);
    Radio::Device::SweepCapture capture;
    U64 segments = 0;
    REQUIRE(device.captureHop(0, samples, segments, capture, [&] { return device.reads < 50; }) ==
            Result::YIELD);
    REQUIRE(segments == 0);
    const auto start = std::chrono::steady_clock::now();
    REQUIRE(device.captureHop(0, samples, segments, capture, [] { return true; }) ==
            Result::TIMEOUT);
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::seconds(4));
    REQUIRE(segments == 0);
}

TEST_CASE("SDR sweeps stop after bounded failures of any hop",
          "[modules][sdr][backend][capture][failure]") {
    CounterDevice device;
    REQUIRE(device.start({.frequency = 100.0e6, .sampleRate = 2.0e6}) == Result::SUCCESS);
    REQUIRE(device.prepareSweep({.frequencies = {100.0e6, 101.0e6, 102.0e6}, .sampleRate = 2.0e6,
                                 .captureLength = 16}) == Result::SUCCESS);
    // The first hop keeps succeeding. Every read of the third hop is a gap.
    device.readLimit = 16;
    for (U64 read = 2; read < 2 + 3 * 9; ++read) {
        device.lossAt.push_back(read);
    }
    U64 captures = 0;
    REQUIRE(device.runSweep([] { return true; }, [&](const Radio::Device::SweepCapture&) {
        ++captures;
        return true;
    }) == Result::ERROR);
    REQUIRE(captures == 2);
    REQUIRE(device.statistics().failedHops == 3);
    REQUIRE(device.tunes == std::vector<F64>{100.0e6, 101.0e6, 102.0e6, 102.0e6, 102.0e6});
}

TEST_CASE("SDR stream buffers never mix generations in a frame",
          "[modules][sdr][backend][capture]") {
    Radio::StreamBuffer buffer;
    REQUIRE(buffer.resize(16) == Result::SUCCESS);
    buffer.reset({.frequency = 100.0e6, .sampleRate = 2.0e6, .generation = 1});
    const std::vector<CF32> samples(12, CF32{1.0f, 0.0f});
    REQUIRE(buffer.push(1, samples.data(), 12));

    std::vector<CF32> frame(8);
    Radio::StreamBuffer::Metadata metadata;
    REQUIRE(buffer.pop(frame.data(), 8, metadata) == Result::SUCCESS);
    REQUIRE(metadata.generation == 1);

    // Four samples of a half-filled frame remain when the settings change.
    buffer.reset({.frequency = 200.0e6, .sampleRate = 2.0e6, .generation = 2});
    REQUIRE(buffer.size() == 0);
    REQUIRE_FALSE(buffer.push(1, samples.data(), 4));
    REQUIRE(buffer.pop(frame.data(), 8, metadata) == Result::INCOMPLETE);
    REQUIRE(buffer.push(2, samples.data(), 8));
    REQUIRE(buffer.pop(frame.data(), 8, metadata) == Result::SUCCESS);
    REQUIRE(metadata.frequency == 200.0e6);
    REQUIRE(buffer.statistics().rejectedSamples == 4);
}

TEST_CASE("SDR stream buffers keep loss totals across resets",
          "[modules][sdr][backend][capture]") {
    Radio::StreamBuffer buffer;
    REQUIRE(buffer.resize(4) == Result::SUCCESS);
    buffer.reset({.generation = 1});
    const std::vector<CF32> samples(6);
    REQUIRE(buffer.push(1, samples.data(), 6));
    buffer.reset({.generation = 2});
    REQUIRE(buffer.push(2, samples.data(), 2));
    const auto stats = buffer.statistics();
    REQUIRE(stats.pushedSamples == 8);
    REQUIRE(stats.overwrittenSamples == 2);
}

TEST_CASE("SDR stream access interrupts captures and applies changes on the caller",
          "[modules][sdr][backend][control]") {
    Radio::StreamAccess access;
    std::atomic<bool> applied{false};
    std::thread::id caller;
    std::thread::id setter;
    auto read = access.lock();
    auto change = std::async(std::launch::async, [&] {
        caller = std::this_thread::get_id();
        return access.change([&] {
            setter = std::this_thread::get_id();
            applied = true;
            return Result::WARNING;
        });
    });
    const bool interrupted = WaitFor([&] { return access.interrupted(); });
    const bool appliedDuringRead = applied;
    read.unlock();
    REQUIRE(change.get() == Result::WARNING);
    REQUIRE(interrupted);
    REQUIRE_FALSE(appliedDuringRead);
    REQUIRE(applied);
    REQUIRE(setter == caller);
    REQUIRE_FALSE(access.interrupted());

    REQUIRE_THROWS(access.change([]() -> Result { throw std::runtime_error("setter failed"); }));
    REQUIRE_FALSE(access.interrupted());
    REQUIRE(access.lock().owns_lock());
}

TEST_CASE("SDR timestamp filter rejects stale epochs, gaps, and guard samples",
          "[modules][sdr][backend][lime]") {
    Radio::TimestampFilter tracker;
    tracker.restart(100, 1000);
    REQUIRE(tracker.epoch() == 1);

    // A restart that did not reset the counter cannot establish a boundary.
    auto decision = tracker.accept(5000, 64);
    REQUIRE(decision.status == Radio::Device::Status::Failure);

    tracker.restart(100, 1000);
    REQUIRE(tracker.epoch() == 2);
    decision = tracker.accept(10, 64);
    REQUIRE(decision.status == Radio::Device::Status::Timeout);
    REQUIRE(tracker.pending() == 36);
    decision = tracker.accept(74, 64);
    REQUIRE(decision.status == Radio::Device::Status::Samples);
    REQUIRE(decision.skip == 36);

    // Status counters can run ahead of the read cursor, so a reported loss
    // rejects this read and everything up to the latest received packet.
    tracker.lose(300);
    decision = tracker.accept(138, 64);
    REQUIRE(decision.status == Radio::Device::Status::Discontinuity);
    REQUIRE(tracker.pending() == 98);
    REQUIRE(tracker.accept(202, 98).status == Radio::Device::Status::Timeout);
    decision = tracker.accept(300, 64);
    REQUIRE(decision.status == Radio::Device::Status::Samples);
    REQUIRE(decision.skip == 0);

    // A later read that starts after a gap is rejected by the timestamp jump.
    decision = tracker.accept(400, 64);
    REQUIRE(decision.status == Radio::Device::Status::Discontinuity);
    REQUIRE(decision.lost == 36);
    REQUIRE(tracker.accept(464, 64).status == Radio::Device::Status::Samples);
    REQUIRE(tracker.epoch() == 2);
}

TEST_CASE("SDR timestamp filter live retunes skip the queued FIFO before their boundary",
          "[modules][sdr][backend][lime][retune]") {
    Radio::TimestampFilter tracker;
    tracker.restart(0, 1000);
    REQUIRE(tracker.accept(0, 100).status == Radio::Device::Status::Samples);

    // The boundary is the latest received packet plus the in-flight margin
    // and the guard. The FIFO before it is drained, not labeled.
    tracker.retune(1000);
    REQUIRE(tracker.epoch() == 1);
    REQUIRE(tracker.pending() == 900);
    auto decision = tracker.accept(100, 600);
    REQUIRE(decision.status == Radio::Device::Status::Timeout);
    REQUIRE(tracker.pending() == 300);
    decision = tracker.accept(700, 400);
    REQUIRE(decision.status == Radio::Device::Status::Samples);
    REQUIRE(decision.skip == 300);
    REQUIRE(tracker.pending() == 0);

    // Loss and FIFO overflow before the first sample of a hop only move the
    // boundary. Nothing of the hop was delivered, so nothing is broken.
    tracker.retune(2000);
    tracker.lose(2500);
    REQUIRE(tracker.pending() == 1400);
    REQUIRE(tracker.accept(1800, 100).status == Radio::Device::Status::Timeout);
    decision = tracker.accept(2400, 200);
    REQUIRE(decision.status == Radio::Device::Status::Samples);
    REQUIRE(decision.skip == 100);
    REQUIRE(tracker.accept(2600, 64).status == Radio::Device::Status::Samples);
}

TEST_CASE("SDR timestamp filter guards reject overflow without accepting earlier samples",
          "[modules][sdr][backend][limits]") {
    const U64 maximum = std::numeric_limits<U64>::max();
    Radio::TimestampFilter tracker;
    tracker.restart(10, maximum);
    REQUIRE(tracker.accept(maximum - 5, 1).status == Radio::Device::Status::Failure);
    REQUIRE(tracker.accept(maximum - 4, 1).status == Radio::Device::Status::Failure);
    tracker.restart(maximum, maximum);
    REQUIRE(tracker.accept(0, 1).status == Radio::Device::Status::Failure);
    tracker.restart(0, maximum);
    REQUIRE(tracker.accept(maximum - 1, 2).status == Radio::Device::Status::Failure);
    tracker.restart(2, maximum);
    REQUIRE(tracker.accept(maximum - 4, 3).status == Radio::Device::Status::Samples);
}

TEST_CASE("SDR timestamp filter tracks retune-only streams without a restart",
          "[modules][sdr][backend][bladerf]") {
    Radio::TimestampFilter tracker;
    tracker.retune(1250);

    auto decision = tracker.accept(1000, 200);
    REQUIRE(decision.status == Radio::Device::Status::Timeout);
    REQUIRE(decision.skip == 200);
    REQUIRE(tracker.pending() == 50);

    decision = tracker.accept(1200, 114);
    REQUIRE(decision.status == Radio::Device::Status::Samples);
    REQUIRE(decision.skip == 50);
    REQUIRE(tracker.pending() == 0);

    REQUIRE(tracker.accept(1314, 20).status == Radio::Device::Status::Samples);
    decision = tracker.accept(1400, 64);
    REQUIRE(decision.status == Radio::Device::Status::Discontinuity);
    REQUIRE(decision.lost == 66);
    REQUIRE(tracker.accept(1464, 64).status == Radio::Device::Status::Samples);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR devices restore the previous settings after a failed change",
                 "[modules][sdr][backend][lifecycle]") {
    std::unique_ptr<Radio::Device> device;
    REQUIRE(Radio::Device::Open(Selector, device) == Result::SUCCESS);
    REQUIRE(device->start({.frequency = 100.0e6, .sampleRate = 2.0e6}) == Result::SUCCESS);
    const U64 generation = device->generation();

    TestScript().failAt = "start:300";
    REQUIRE(device->apply({.frequency = 300.0e6, .sampleRate = 4.0e6}) == Result::ERROR);
    REQUIRE(device->applied().frequency == 100.0e6);
    REQUIRE(device->applied().sampleRate == 2.0e6);
    REQUIRE(device->generation() > generation);
    const auto calls = TestScript().calls();
    REQUIRE(std::count(calls.begin(), calls.end(), "start:300") == 1);
    REQUIRE(calls.back() == "start:100");

    TestScript().failAt.clear();
    REQUIRE(device->apply({.frequency = 300.0e6, .sampleRate = 4.0e6}) == Result::SUCCESS);
    REQUIRE(device->applied().frequency == 300.0e6);
    REQUIRE(device->applied().sampleRate == 4.0e6);
}

//
// Module.
//

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR module rejects invalid configurations before opening hardware",
                 "[modules][sdr][validation]") {
    const auto check = [](const Modules::Sdr& config) {
        const auto module = BuildSdrModule();
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE(module->state() == Module::State::ERRORED);
        REQUIRE(module->outputs().empty());
    };

    auto config = ScriptedConfig();
    config.frequency = std::numeric_limits<F32>::quiet_NaN();
    check(config);
    config = ScriptedConfig();
    config.sampleRate = 0.0f;
    check(config);
    config = ScriptedConfig();
    config.numberOfBatches = 0;
    check(config);
    config = ScriptedConfig();
    config.numberOfBatches = std::numeric_limits<U64>::max();
    config.numberOfTimeSamples = 2;
    check(config);
    config = ScriptedConfig();
    config.bufferMultiplier = std::numeric_limits<U64>::max();
    check(config);
    REQUIRE(TestScript().opens == 0);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR module stays incomplete without a selected or connected device",
                 "[modules][sdr][lifecycle]") {
    auto config = ScriptedConfig();
    config.deviceString.clear();
    auto module = BuildSdrModule();
    REQUIRE(module->create("test", config, {}) == Result::INCOMPLETE);
    REQUIRE(module->destroy() == Result::SUCCESS);

    config.deviceString = "driver=scripted,serial=missing";
    module = BuildSdrModule();
    REQUIRE(module->create("test", config, {}) == Result::INCOMPLETE);
    REQUIRE(module->destroy() == Result::SUCCESS);
    REQUIRE(TestScript().opens == 0);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR module publishes the applied settings with its output layout",
                 "[modules][sdr][lifecycle]") {
    TestScript().rateOffset = 125.0;
    auto config = ScriptedConfig();
    config.numberOfBatches = 3;
    config.numberOfTimeSamples = 17;
    const auto module = BuildSdrModule();
    const ModuleCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);

    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{3, 17});
    SignalAxes axes;
    REQUIRE(ResolveSignalAxes(output, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{1});
    REQUIRE(axes.batch == Index{0});
    REQUIRE(AttributeOf(output, "frequency") == 100.0e6f);
    REQUIRE(AttributeOf(output, "sampleRate") == static_cast<F32>(2.0e6 + 125.0));
    REQUIRE(module->getImpl<Modules::SdrImpl>()->listAntennas() ==
            std::vector<std::string>{"RX1", "RX2"});
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR retuning never relabels samples captured before the change",
                 "[modules][sdr][metadata][retune]") {
    TestScript().budget = 12;
    const auto module = BuildSdrModule();
    const ModuleCleanup cleanup{module};
    REQUIRE(module->create("test", ScriptedConfig(), {}) == Result::SUCCESS);
    auto* sdr = module->getImpl<Modules::SdrImpl>();
    const auto& stream = sdr->*SdrImplAccess::streamMember();
    const auto runtime = RuntimeOf(module);
    const auto& output = module->outputs().at("signal").tensor;

    REQUIRE(WaitFor([&] { return stream.size() == 12; }));
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    REQUIRE(output.data<CF32>()[0].real() == 100.0f);
    REQUIRE(AttributeOf(output, "frequency") == 100.0e6f);

    // Four samples of a partial frame remain. Applying the retune neither
    // relabels them nor lets them complete a frame with the new settings.
    REQUIRE(stream.size() == 4);
    REQUIRE(module->reconfigure({{"frequency", 250.0e6f}}) == Result::SUCCESS);
    REQUIRE(WaitFor([&] { return stream.size() == 0 && stream.metadata().frequency == 250.0e6; }));
    REQUIRE(runtime->computeSubmit() == Result::YIELD);
    REQUIRE(AttributeOf(output, "frequency") == 100.0e6f);

    TestScript().budget = 8;
    REQUIRE(WaitFor([&] { return stream.size() == 8; }));
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    REQUIRE(AttributeOf(output, "frequency") == 250.0e6f);
    for (U64 i = 0; i < output.size(); ++i) {
        REQUIRE(output.data<CF32>()[i].real() == 250.0f);
    }
    REQUIRE(static_cast<const Modules::Sdr&>(module->config()).frequency == 250.0e6f);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR failed retunes keep the previous labels",
                 "[modules][sdr][metadata][retune]") {
    TestScript().budget = 0;
    const auto module = BuildSdrModule();
    const ModuleCleanup cleanup{module};
    REQUIRE(module->create("test", ScriptedConfig(), {}) == Result::SUCCESS);
    TestScript().failAt = "tune:300";
    REQUIRE(module->reconfigure({{"frequency", 300.0e6f}}) == Result::ERROR);
    REQUIRE(static_cast<const Modules::Sdr&>(module->config()).frequency == 100.0e6f);
    REQUIRE(module->getImpl<Modules::SdrImpl>()->getApplied().frequency == 100.0e6);

    // Out-of-range values only warn and leave the device untouched.
    const auto calls = TestScript().calls();
    REQUIRE(module->reconfigure({{"frequency", 5000.0e6f}}) == Result::WARNING);
    REQUIRE(TestScript().calls() == calls);

    TestScript().failAt.clear();
    TestScript().budget = 8;
    const auto& stream = module->getImpl<Modules::SdrImpl>()->*SdrImplAccess::streamMember();
    REQUIRE(WaitFor([&] { return stream.size() == 8; }));
    REQUIRE(RuntimeOf(module)->computeSubmit() == Result::SUCCESS);
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(AttributeOf(output, "frequency") == 100.0e6f);
    REQUIRE(output.data<CF32>()[0].real() == 100.0f);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR frequency edits only retune and other edits only touch what changed",
                 "[modules][sdr][retune][latency]") {
    TestScript().budget = 0;
    auto config = ScriptedConfig();
    config.biasTee = true;
    config.automaticGain = false;
    const auto module = BuildSdrModule();
    const ModuleCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    auto* sdr = module->getImpl<Modules::SdrImpl>();
    const auto& stream = sdr->*SdrImplAccess::streamMember();
    const auto calls = [&] {
        const auto all = TestScript().calls();
        const auto created = std::find(all.begin(), all.end(), "biastee:true");
        return std::vector<std::string>(created + 1, all.end());
    };
    REQUIRE(calls().empty());

    // A frequency edit uses only the tuning path. Sample rate, gain, Bias-T,
    // and the stream setup stay as they were.
    const U64 generation = stream.metadata().generation;
    REQUIRE(module->reconfigure({{"frequency", 250.0e6f}}) == Result::SUCCESS);
    REQUIRE(WaitFor([&] { return sdr->getApplied().frequency == 250.0e6; }));
    REQUIRE(calls() == std::vector<std::string>{"tune:250"});
    const auto applied = sdr->getApplied();
    REQUIRE(applied.frequency == 250.0e6);
    REQUIRE(applied.sampleRate == 2.0e6);
    REQUIRE_FALSE(applied.automaticGain);
    REQUIRE(applied.biasTee);
    REQUIRE(stream.metadata().generation > generation);
    REQUIRE(stream.metadata().frequency == 250.0e6);

    // Unchanged values and block-only settings never reach the radio.
    REQUIRE(module->reconfigure({{"frequency", 250.0e6f}}) == Result::SUCCESS);
    REQUIRE(module->reconfigure({{"frequency", 250.0e6f}, {"sampleRate", 2.0e6f}}) ==
            Result::SUCCESS);
    REQUIRE(calls() == std::vector<std::string>{"tune:250"});

    // Rate and gain changes need the full configuration path.
    REQUIRE(module->reconfigure({{"sampleRate", 4.0e6f}}) == Result::SUCCESS);
    REQUIRE(WaitFor([&] { return sdr->getApplied().sampleRate == 4.0e6; }));
    REQUIRE(calls() == std::vector<std::string>{"tune:250", "stop", "start:250"});
    REQUIRE(sdr->getApplied().sampleRate == 4.0e6);
    REQUIRE(sdr->getApplied().frequency == 250.0e6);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR live edits finish synchronously and failed transactions remain retryable",
                 "[modules][sdr][reconfigure][control]") {
    TestScript().budget = 0;
    TestScript().readDelay = std::chrono::milliseconds(10);
    const auto module = BuildSdrModule();
    const ModuleCleanup cleanup{module};
    auto config = ScriptedConfig();
    config.automaticGain = false;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    REQUIRE(WaitFor([] { return TestScript().activeReads > 0; }));
    auto* sdr = module->getImpl<Modules::SdrImpl>();

    Parser::Map edit;
    std::string failure;
    SECTION("frequency") {
        edit = {{"frequency", 300.0e6f}};
        failure = "tune:300";
    }
    SECTION("sample rate") {
        edit = {{"sampleRate", 4.0e6f}};
        edit["frequency"] = 300.0e6f;
        failure = "start:300";
    }
    SECTION("automatic gain") {
        edit = {{"automaticGain", true}};
        failure = "mode:true";
    }
    SECTION("manual gain after a successful retune") {
        edit = {{"frequency", 300.0e6f}, {"manualGain", 40.0f}};
        failure = "gain:40";
    }
    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().failAt = failure;
    }
    REQUIRE(module->reconfigure(edit) == Result::ERROR);
    REQUIRE(sdr->getApplied().frequency == config.frequency);
    REQUIRE(sdr->getApplied().sampleRate == config.sampleRate);
    REQUIRE(sdr->getApplied().automaticGain == config.automaticGain);
    REQUIRE(sdr->getApplied().gain == config.manualGain);
    Parser::Map before;
    Parser::Map retained;
    REQUIRE(config.serialize(before) == Result::SUCCESS);
    REQUIRE(module->config(retained) == Result::SUCCESS);
    REQUIRE(retained == before);

    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().failAt.clear();
    }
    REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
    REQUIRE(config.deserialize(edit) == Result::SUCCESS);
    REQUIRE(sdr->getApplied().frequency == config.frequency);
    REQUIRE(sdr->getApplied().sampleRate == config.sampleRate);
    REQUIRE(sdr->getApplied().automaticGain == config.automaticGain);
    REQUIRE(sdr->getApplied().gain == config.manualGain);
    REQUIRE(TestScript().concurrentChanges == 0);
}

TEST_CASE_METHOD(ScriptedFlowgraph,
                 "SDR block-only edits leave the radio untouched",
                 "[modules][sdr][block][retune][latency]") {
    TestScript().budget = 0;
    Blocks::Sdr config;
    config.deviceString = Selector;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    const auto calls = TestScript().calls();
    REQUIRE(flowgraph->blockReconfigure("radio", {{"frequencyStep", 25000.0f}}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    REQUIRE(TestScript().calls() == calls);
    REQUIRE(flowgraph->blockReconfigure("radio", {{"frequency", 101.1e6f}}) == Result::SUCCESS);
    auto expected = calls;
    expected.push_back("tune:101");
    REQUIRE(WaitFor([&] { return TestScript().calls() == expected; }));
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR startup failures release the device and its antenna power",
                 "[modules][sdr][lifecycle][bias-tee]") {
    auto config = ScriptedConfig();
    config.biasTee = true;

    SECTION("normal lifecycle disables Bias-T on teardown") {
        const auto module = BuildSdrModule();
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(TestScript().calls() == std::vector<std::string>{
            "open", "start:100", "biastee:true", "stop", "biastee:false", "close",
        });
    }

    SECTION("failed startup closes the device") {
        TestScript().failAt = "start";
        const auto module = BuildSdrModule();
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE(TestScript().calls() == std::vector<std::string>{"open", "start:100", "close"});
    }

    SECTION("repeated open and close cycles reuse the device") {
        const auto module = BuildSdrModule();
        for (int i = 0; i < 3; ++i) {
            REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
        REQUIRE(TestScript().opens == 3);
        const auto calls = TestScript().calls();
        REQUIRE(std::count(calls.begin(), calls.end(), "close") == 3);
        REQUIRE(std::count(calls.begin(), calls.end(), "biastee:false") == 3);
    }
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR receive failures reach the scheduler",
                 "[modules][sdr][receive]") {
    TestScript().budget = 0;
    const auto module = BuildSdrModule();
    const ModuleCleanup cleanup{module};
    REQUIRE(module->create("test", ScriptedConfig(), {}) == Result::SUCCESS);
    REQUIRE(module->context()->scheduler()->hasPendingCompute() == Result::TIMEOUT);
    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().nextRead = Radio::Device::Capture{.status = Radio::Device::Status::Disconnected,
                                               .error = "unplugged"};
    }
    REQUIRE(WaitFor([&] {
        return module->context()->scheduler()->hasPendingCompute() == Result::ERROR;
    }));
    REQUIRE(RuntimeOf(module)->computeSubmit() == Result::ERROR);
}

TEST_CASE_METHOD(ScriptedFlowgraph,
                 "SDR loss bar separates buffer overwrites from device gaps",
                 "[modules][sdr][metrics]") {
    TestScript().budget = 0;
    Blocks::Sdr config;
    config.deviceString = Selector;
    config.frequency = 100.0e6f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    config.bufferMultiplier = 1;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);

    const auto metric = [&](const std::string& name) {
        std::vector<Flowgraph::View::MetricEntry> metrics;
        REQUIRE(flowgraph->view().metrics("radio", metrics) == Result::SUCCESS);
        const auto it = std::find_if(metrics.begin(), metrics.end(), [&](const auto& entry) {
            return entry.name == name;
        });
        REQUIRE(it != metrics.end());
        REQUIRE(it->format == Parser::Map{{"type", "progressbar"}});
        return std::any_cast<std::pair<std::string, F32>>(it->value);
    };

    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().nextRead = Radio::Device::Capture{.status = Radio::Device::Status::Discontinuity, .lost = 64};
    }
    REQUIRE(WaitFor([&] { return metric("bufferLoss").first == "0.00% (1 GAP)"; }));
    REQUIRE(metric("bufferLoss").second == 0.0f);

    // Twenty samples into an eight-sample buffer overwrite twelve.
    TestScript().budget = 20;
    REQUIRE(WaitFor([&] { return TestScript().budget == 0; }));
    REQUIRE(WaitFor([&] { return metric("bufferLoss").second > 0.0f; }));
    REQUIRE(metric("bufferLoss") == std::pair<std::string, F32>{"60.00% (1 GAP)", 0.6f});
    REQUIRE(metric("throughput").first.ends_with("/ 16.0 MB/s"));
}

TEST_CASE_METHOD(ScriptedFlowgraph,
                 "SDR blocks list fixed sample rates even when the radio rejects the rate",
                 "[modules][sdr][block][selection]") {
    TestScript().budget = 0;
    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().sampleRates = {{10.0e6, 10.0e6}, {2.5e6, 2.5e6}};
        TestScript().failAt = "start";
    }
    Blocks::Sdr config;
    config.deviceString = Selector;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Errored);

    const auto format = [&] {
        const auto view = viewBlock("radio");
        const auto field = std::find_if(view.interfaceConfigs.begin(), view.interfaceConfigs.end(),
                                        [](const auto& entry) { return entry.name == "sampleRate"; });
        REQUIRE(field != view.interfaceConfigs.end());
        return field->format;
    };
    REQUIRE(Parser::Get<std::string>(format(), "type") == "dropdown");
    std::vector<std::pair<std::string, std::string>> rates;
    for (const auto& option : Parser::Get<std::vector<Parser::Map>>(format(), "options")) {
        rates.emplace_back(Parser::Get<std::string>(option, "label"),
                           Parser::Get<std::string>(option, "value"));
    }
    REQUIRE(rates == std::vector<std::pair<std::string, std::string>>{
        {"2.5 MHz", "2500000"},
        {"10 MHz", "1e+07"},
    });

    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().failAt.clear();
    }
    REQUIRE(flowgraph->blockReconfigure("radio", {{"sampleRate", std::string("1e+07")}}) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    REQUIRE(Parser::Get<F32>(viewBlock("radio").config, "sampleRate") == 10.0e6f);
    REQUIRE(Parser::Get<std::string>(format(), "type") == "dropdown");

    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().sampleRates.clear();
    }
    REQUIRE(flowgraph->blockReconfigure("radio", {{"antenna", std::string("RX2")}}) ==
            Result::SUCCESS);
    REQUIRE(Parser::Get<std::string>(format(), "type") == "float");
}

TEST_CASE_METHOD(ScriptedFlowgraph,
                 "SDR blocks list direct devices and keep saved selections",
                 "[modules][sdr][block][selection]") {
    Blocks::Sdr config;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    auto block = viewBlock("radio");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(TestScript().opens == 0);

    const auto options = [&] {
        const auto view = viewBlock("radio");
        const auto field = std::find_if(view.interfaceConfigs.begin(), view.interfaceConfigs.end(),
                                        [](const auto& entry) { return entry.name == "deviceString"; });
        REQUIRE(field != view.interfaceConfigs.end());
        return Parser::Get<std::vector<Parser::Map>>(field->format, "options");
    };
    REQUIRE(options() == std::vector<Parser::Map>{
        {{"label", "None"}, {"value", ""}},
        {{"label", "Scripted Radio"}, {"value", Selector}},
    });

    REQUIRE(flowgraph->blockReconfigure("radio", {{"deviceString", Selector}}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    const auto saved = viewBlock("radio").config;
    REQUIRE(Parser::Get<std::string>(saved, "deviceString") == Selector);

    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().devices.clear();
    }
    Radio::Device::ClearCache();
    REQUIRE(flowgraph->blockRecreate("radio", saved) == Result::SUCCESS);
    block = viewBlock("radio");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(block.diagnostic.find("not connected") != std::string::npos);
    REQUIRE(options().back() == Parser::Map{{"label", "Configured device"}, {"value", Selector}});
    REQUIRE(Parser::Get<std::string>(block.config, "deviceString") == Selector);

    REQUIRE(flowgraph->blockReconfigure("radio", {{"deviceString", ""}}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Incomplete);
    REQUIRE(TestScript().opens == 1);
}
