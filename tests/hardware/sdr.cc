#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <exception>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "domains/io/sdr/devices/device.hh"
#include "domains/io/sdr/devices/stream.hh"
#include "domains/io/sdr/devices/sweep.hh"
#include "domains/io/sdr_sweep/tools/sweep.hh"
#include "jetstream/logger.hh"

using namespace Jetstream;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Plan = Modules::Sweep::Plan;
using Radio::Device;
using Radio::Stream;
using Radio::Sweep;

namespace {

constexpr F64 RateTolerance = 0.30;
constexpr F64 LostFraction = 0.01;
constexpr F64 GapsPerSecond = 2.0;
constexpr auto Warmup = 1s;
constexpr auto Measurement = 3s;

void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void Success(const Result result, const std::string& operation) {
    Require(result == Result::SUCCESS, jst::fmt::format("{} failed", operation));
}

bool Near(const F64 actual, const F64 expected, const F64 relative = 1e-4) {
    return std::isfinite(actual) && std::abs(actual - expected) <=
        std::max(1.0, std::abs(expected) * relative);
}

struct Results {
    U64 passed = 0;
    U64 failed = 0;

    template<typename Function>
    void run(const std::string& name, Function function) {
        jst::fmt::print("[RUN] {}\n", name);
        try {
            function();
            ++passed;
            jst::fmt::print("[PASS] {}\n", name);
        } catch (const std::exception& error) {
            ++failed;
            jst::fmt::print("[FAIL] {}: {}\n", name, error.what());
        }
    }
};

void Ranges(const std::vector<Device::Range>& ranges, const std::string& name) {
    Require(!ranges.empty(), jst::fmt::format("{} ranges are missing", name));
    for (const auto& range : ranges) {
        Require(std::isfinite(range.minimum) && std::isfinite(range.maximum) &&
                range.minimum > 0.0 && range.minimum <= range.maximum,
                jst::fmt::format("{} range is invalid", name));
    }
}

void Capabilities(const Device::Capabilities& caps) {
    Ranges(caps.frequencyRanges, "frequency");
    Ranges(caps.sampleRateRanges, "sample rate");
    Require(Device::Range::Contains(caps.sampleRateRanges, caps.sweepSampleRate),
            "default sweep rate is outside the advertised ranges");
    Require(std::isfinite(caps.usableBandwidth) && caps.usableBandwidth > 0.0 &&
            caps.usableBandwidth <= 1.0, "invalid usable bandwidth");
}

void GainRange(const std::optional<Device::Range>& range) {
    Require(range && std::isfinite(range->minimum) && std::isfinite(range->maximum) &&
            range->minimum <= range->maximum, "gain range is missing or invalid");
}

Device::Range FrequencyBand(const Device::Capabilities& caps) {
    Ranges(caps.frequencyRanges, "frequency");
    return *std::max_element(caps.frequencyRanges.begin(), caps.frequencyRanges.end(),
                            [](const auto& a, const auto& b) {
        return a.maximum - a.minimum < b.maximum - b.minimum;
    });
}

std::vector<F64> Rates(const Device::Capabilities& caps) {
    Capabilities(caps);
    std::set<F64> rates{caps.sweepSampleRate};
    for (const auto& range : caps.sampleRateRanges) {
        rates.insert(range.minimum);
        rates.insert(std::round((range.minimum + range.maximum) / 2.0));
        rates.insert(range.maximum);
    }
    std::vector<F64> result;
    for (const auto rate : rates) {
        if (Device::Range::Contains(caps.sampleRateRanges, rate)) {
            result.push_back(rate);
        }
    }
    return result;
}

std::unique_ptr<Stream> OpenStream(const Device::Info& info) {
    std::unique_ptr<Stream> stream;
    Success(Stream::Open(info.selector(), stream), "open stream");
    Require(stream != nullptr, "open returned no stream");
    return stream;
}

std::unique_ptr<Sweep> OpenSweep(const Device::Info& info) {
    std::unique_ptr<Sweep> sweep;
    Success(Sweep::Open(info.selector(), sweep), "open sweep");
    Require(sweep != nullptr, "open returned no sweep");
    return sweep;
}

Stream::Settings Settings(const Device::Capabilities& caps) {
    const auto band = FrequencyBand(caps);
    Stream::Settings settings{
        .frequency = std::round(band.minimum + (band.maximum - band.minimum) * 0.4),
        .sampleRate = caps.sweepSampleRate,
        .automaticGain = false,
    };
    if (!caps.antennas.empty()) {
        const auto automatic = std::find(caps.antennas.begin(), caps.antennas.end(), "Auto");
        settings.antenna = automatic != caps.antennas.end() ? *automatic : caps.antennas.front();
    }
    return settings;
}

void Applied(const Stream& stream, const Stream::Settings& settings) {
    const auto& applied = stream.applied();
    Require(Near(applied.frequency, settings.frequency), "applied frequency mismatch");
    Require(Near(applied.sampleRate, settings.sampleRate), "applied sample rate mismatch");
    Require(applied.automaticGain == settings.automaticGain, "applied AGC mismatch");
    GainRange(applied.gainRange);
    if (!settings.automaticGain) {
        Require(Near(applied.gain, stream.resolveGain(settings.gain, applied.gainRange)),
                "applied gain mismatch");
    }
}

struct Delivery {
    U64 samples = 0;
    U64 gaps = 0;
    U64 lost = 0;
    U64 timestampGaps = 0;
    U64 timestampMissing = 0;
    F64 seconds = 0.0;
    F64 cpuPercent = 0.0;
    F64 firstSampleSeconds = 0.0;
};

Delivery Receive(Stream& stream, const Clock::duration duration) {
    std::vector<CF32> buffer(65536);
    Delivery delivery;
    std::optional<U64> expected;
    U64 epoch = 0;
    const auto begin = Clock::now();
    const auto cpuBegin = std::clock();
    auto progress = begin;
    while (Clock::now() - begin < duration || delivery.samples == 0) {
        const auto capture = stream.read(buffer, 100ms);
        Require(capture.status != Device::Status::Failure &&
                capture.status != Device::Status::Disconnected,
                jst::fmt::format("receive: {}", capture.error));
        delivery.lost += capture.lost;
        if (capture.status == Device::Status::Discontinuity) {
            ++delivery.gaps;
            expected.reset();
        }
        if (capture.status == Device::Status::Samples && capture.count > 0) {
            Require(capture.count <= buffer.size(), "capture exceeds receive buffer");
            Require(capture.generation == stream.generation(), "stale capture generation");
            Require(Near(capture.frequency, stream.applied().frequency), "capture frequency mismatch");
            Require(Near(capture.sampleRate, stream.applied().sampleRate), "capture rate mismatch");
            if (capture.timestamp) {
                if (expected && (epoch != capture.epoch || *expected != *capture.timestamp)) {
                    ++delivery.timestampGaps;
                    if (epoch == capture.epoch && *capture.timestamp > *expected) {
                        delivery.timestampMissing += *capture.timestamp - *expected;
                    }
                }
                expected = *capture.timestamp + capture.count;
                epoch = capture.epoch;
            } else {
                expected.reset();
            }
            for (U64 i = 0; i < capture.count; i += 257) {
                Require(std::isfinite(buffer[i].real()) && std::isfinite(buffer[i].imag()),
                        "nonfinite IQ sample");
            }
            if (delivery.samples == 0) {
                delivery.firstSampleSeconds = std::chrono::duration<F64>(Clock::now() - begin).count();
            }
            delivery.samples += capture.count;
            progress = Clock::now();
        }
        Require(Clock::now() - progress < 3s, "stream stalled for three seconds");
    }
    delivery.seconds = std::chrono::duration<F64>(Clock::now() - begin).count();
    delivery.cpuPercent = 100.0 * (std::clock() - cpuBegin) / CLOCKS_PER_SEC / delivery.seconds;
    Require(delivery.samples > 0, "no samples delivered");
    return delivery;
}

void Throughput(const Device::Info& info, const F64 rate) {
    auto stream = OpenStream(info);
    auto settings = Settings(stream->capabilities());
    settings.sampleRate = rate;
    Success(stream->start(settings), "start stream");
    Applied(*stream, settings);
    Receive(*stream, Warmup);
    const auto delivery = Receive(*stream, Measurement);
    const F64 deliveredRate = delivery.samples / delivery.seconds;
    jst::fmt::print("  requested={:.3f} applied={:.3f} delivered={:.3f} MS/s, gaps={} "
                    "timestamp_gaps={} timestamp_missing={} reported_lost={} samples={} "
                    "seconds={:.3f} cpu={:.1f}%\n",
                    rate / 1e6, stream->applied().sampleRate / 1e6, deliveredRate / 1e6,
                    delivery.gaps, delivery.timestampGaps, delivery.timestampMissing, delivery.lost,
                    delivery.samples, delivery.seconds, delivery.cpuPercent);
    Require(std::abs(deliveredRate / rate - 1.0) <= RateTolerance,
            "delivered throughput is outside +/-30% of requested rate");
    Require(delivery.gaps + delivery.timestampGaps <=
            std::max(1.0, GapsPerSecond * delivery.seconds), "more than two gaps per second");
    Require(std::max(delivery.lost, delivery.timestampMissing) <= rate * delivery.seconds * LostFraction,
            "reported loss exceeds 1% of requested samples");
}

void Controls(const Device::Info& info) {
    auto stream = OpenStream(info);
    const auto& caps = stream->capabilities();
    auto settings = Settings(caps);
    Success(stream->start(settings), "start controls stream");
    Applied(*stream, settings);
    Receive(*stream, Warmup);
    const auto change = [&](const std::string& name) {
        Success(stream->update(settings), name);
        Applied(*stream, settings);
        const auto delivery = Receive(*stream, 1s);
        jst::fmt::print("  {}: samples={}, first IQ={:.3f} s, generation={}, gaps={}, reported_lost={}\n",
                        name, delivery.samples, delivery.firstSampleSeconds, stream->generation(),
                        delivery.gaps, delivery.lost);
    };
    const auto band = FrequencyBand(caps);
    settings.frequency = std::round(band.minimum + (band.maximum - band.minimum) * 0.6);
    const auto generation = stream->generation();
    change("frequency update");
    Require(stream->generation() > generation, "retune did not advance generation");
    const auto range = *stream->applied().gainRange;
    for (const F64 fraction : {0.25, 0.75}) {
        settings.gain = range.minimum + fraction * (range.maximum - range.minimum);
        change("gain update");
    }
    if (caps.automaticGain) {
        settings.automaticGain = true;
        change("AGC on");
        settings.automaticGain = false;
        change("AGC off");
    } else {
        jst::fmt::print("  [SKIP] AGC is not supported\n");
    }
    for (const auto rate : Rates(caps)) {
        if (rate != settings.sampleRate) {
            settings.sampleRate = rate;
            change(jst::fmt::format("sample-rate update to {:.3f} MS/s", rate / 1e6));
        }
    }
    settings.sampleRate = Rates(caps).front();
    change("lowest-rate restart preparation");
    Success(stream->start(settings), "stop/restart at lowest rate");
    Applied(*stream, settings);
    Receive(*stream, Warmup);
    settings.sampleRate = caps.sweepSampleRate;
    change("restore default rate");
    Success(stream->start(settings), "stop/restart stream");
    Applied(*stream, settings);
    Receive(*stream, Warmup);
}

void SweepRun(Sweep& sweep, const Plan& plan, Sweep::Settings& settings, const bool controls) {
    Modules::Sweep::Assembler assembler;
    Success(assembler.configure(plan), "configure sweep assembler");
    std::vector<CF32> output(plan.outputBins);
    std::vector<U64> captured(plan.hops.size());
    U64 completed = 0;
    U64 captures = 0;
    bool changed = false;
    std::optional<Clock::time_point> measured;
    std::clock_t cpuBegin = 0;
    Clock::time_point finished;
    const auto begin = Clock::now();
    const auto before = sweep.statistics().failedHops;
    const auto consume = [&](const Sweep::Capture& capture) {
        Require(capture.hop < plan.hops.size(), "invalid sweep hop index");
        Require(Near(capture.frequency, plan.hops[capture.hop].frequency), "sweep frequency mismatch");
        Require(capture.samples.size() == plan.captureSamples(), "incorrect sweep capture length");
        ++captures;
        ++captured[capture.hop];
        Success(assembler.add(capture), "assemble capture");
        if (!assembler.complete()) {
            return true;
        }
        Success(assembler.synthesize(output), "synthesize sweep");
        Require(std::all_of(output.begin(), output.end(), [](const auto& sample) {
            return std::isfinite(sample.real()) && std::isfinite(sample.imag());
        }), "nonfinite sweep output");
        assembler.begin();
        if (measured) {
            ++completed;
        } else {
            measured = Clock::now();
            cpuBegin = std::clock();
        }
        if (controls && !changed) {
            GainRange(sweep.applied().gainRange);
            const auto range = *sweep.applied().gainRange;
            settings.gain = range.minimum + 0.75 * (range.maximum - range.minimum);
            if (plan.mode == Plan::Mode::Retune) {
                settings.settlingTime = 0.01;
            }
            Success(sweep.update(settings), "live sweep gain/settling update");
            Require(Near(sweep.applied().gain,
                         sweep.resolveGain(settings.gain, sweep.applied().gainRange)),
                    "sweep gain mismatch after update");
            Require(sweep.settings().settlingTime == settings.settlingTime,
                    "sweep settling setting mismatch");
            changed = true;
        }
        finished = Clock::now();
        return completed < 2 || finished - *measured < Measurement;
    };
    std::exception_ptr failure;
    const auto result = sweep.run([&] { return Clock::now() - begin < 15s; },
                                 [&](const auto& capture) {
        try {
            return consume(capture);
        } catch (...) {
            failure = std::current_exception();
            return false;
        }
    });
    if (failure) {
        std::rethrow_exception(failure);
    }
    Success(result, "run sweep");
    Require(measured && completed >= 2, "sweep stalled or did not finish two measured spans");
    Require(std::all_of(captured.begin(), captured.end(), [](const auto count) { return count >= 3; }),
            "not every hop was captured in warmup and measured sweeps");
    Require(!controls || changed, "live sweep controls were not exercised");
    const F64 seconds = std::chrono::duration<F64>(finished - *measured).count();
    const F64 refresh = completed / seconds;
    const auto failed = sweep.statistics().failedHops - before;
    const F64 cpuPercent = 100.0 * (std::clock() - cpuBegin) / CLOCKS_PER_SEC / seconds;
    const auto settling = plan.mode == Plan::Mode::Firmware ? "firmware-fixed" :
        settings.settlingTime < 0.0 ? "driver-default" :
        jst::fmt::format("{:.3f} ms", settings.settlingTime * 1e3);
    jst::fmt::print("  {}: {:.3f} MHz, {} hops, {} samples/FFT, dwell={}, settling={}, "
                    "{:.3f} GHz/s ({:.3f} Hz), sweeps={} failed_hops={}, cpu={:.1f}%\n",
                    Plan::ModeName(plan.mode), plan.span() / 1e6, plan.hops.size(),
                    plan.segmentLength, plan.dwell, settling, plan.span() / 1e9 * refresh,
                    refresh, completed, failed, cpuPercent);
    jst::fmt::print("  sweep packet/sample loss: unavailable through the generic Sweep API\n");
    Require(failed <= std::max<U64>(1, captures / 100), "more than 1% failed sweep hops");
}

void Sweeps(const Device::Info& info, const Plan::Mode mode) {
    auto sweep = OpenSweep(info);
    const auto& caps = sweep->capabilities();
    Capabilities(caps);
    if (mode == Plan::Mode::Firmware && !caps.firmwareSweep) {
        jst::fmt::print("  [SKIP] firmware sweep is not advertised by this device/firmware\n");
        return;
    }
    const auto base = Settings(caps);
    const auto band = FrequencyBand(caps);
    const F64 rate = mode == Plan::Mode::Firmware ? caps.firmwareSweep->sampleRate : caps.sweepSampleRate;
    const F64 span = std::min(4.0 * rate, (band.maximum - band.minimum) / 4.0);
    Plan::Request request{
        .startFrequency = base.frequency,
        .endFrequency = base.frequency + span,
        .sampleRate = rate,
        .resolution = rate / 2048.0,
        .dwell = 2,
        .mode = mode,
    };
    Plan plan;
    Success(Plan::Create(request, caps, plan), "create multi-hop plan");
    Sweep::Settings settings{
        .sampleRate = plan.sampleRate,
        .antenna = base.antenna,
        .automaticGain = false,
        .gain = base.gain,
    };
    Success(sweep->configure(settings), "configure sweep");
    GainRange(sweep->applied().gainRange);
    Require(Near(sweep->applied().sampleRate, rate), "sweep applied rate mismatch");
    request.sampleRate = sweep->applied().sampleRate;
    Success(Plan::Create(request, caps, plan), "replan at applied sample rate");
    Require(plan.hops.size() > 1, "plan is not multi-hop");
    Success(sweep->start(plan.deviceRequest()), "start sweep");
    SweepRun(*sweep, plan, settings, false);
    SweepRun(*sweep, plan, settings, true);
    Success(sweep->start(plan.deviceRequest()), "stop/restart sweep");
    SweepRun(*sweep, plan, settings, false);
}

void SweepGains(const Device::Info& info) {
    auto stream = OpenStream(info);
    const auto base = Settings(stream->capabilities());
    const auto band = FrequencyBand(stream->capabilities());
    auto streamSettings = base;
    Success(stream->start(streamSettings), "start gain-range reference stream");
    std::vector<F64> frequencies;
    std::vector<Device::Range> ranges;
    for (const F64 position : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        streamSettings.frequency = std::round(band.minimum + position * (band.maximum - band.minimum));
        Success(stream->update(streamSettings), "tune gain-range reference");
        GainRange(stream->applied().gainRange);
        frequencies.push_back(streamSettings.frequency);
        ranges.push_back(*stream->applied().gainRange);
    }
    const F64 requestedGain = std::min_element(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return a.minimum < b.minimum;
    })->minimum;
    std::vector<F64> gains;
    for (const auto& range : ranges) {
        gains.push_back(stream->resolveGain(requestedGain, range));
    }
    stream.reset();

    auto sweep = OpenSweep(info);
    Success(sweep->configure({.sampleRate = base.sampleRate, .antenna = base.antenna,
                              .automaticGain = false, .gain = requestedGain}), "configure cross-band sweep");
    const Sweep::Request request{
        .frequencies = frequencies,
        .sampleRate = sweep->applied().sampleRate,
        .captureLength = 2048,
    };
    Success(sweep->start(request), "start cross-band sweep");
    U64 captures = 0;
    const auto deadline = Clock::now() + 15s;
    Success(sweep->run([&] { return Clock::now() < deadline; }, [&](const auto& capture) {
        Require(capture.hop == captures % frequencies.size(), "cross-band hop mismatch");
        const auto& applied = sweep->applied();
        GainRange(applied.gainRange);
        const auto& expected = ranges[capture.hop];
        Require(std::abs(applied.gainRange->minimum - expected.minimum) < 1e-6 &&
                std::abs(applied.gainRange->maximum - expected.maximum) < 1e-6,
                "sweep/stream gain-range mismatch");
        Require(Near(applied.gain, gains[capture.hop]), "cross-band applied gain mismatch");
        if (captures < frequencies.size()) {
            jst::fmt::print("  {:.3f} MHz: gain range [{:.1f}, {:.1f}], applied gain {:.1f}\n",
                            capture.frequency / 1e6, applied.gainRange->minimum,
                            applied.gainRange->maximum, applied.gain);
        }
        return ++captures < frequencies.size() * 2;
    }), "capture cross-band sweep");
    Require(captures == frequencies.size() * 2, "cross-band sweep stalled");
    jst::fmt::print("  cross-band sweep: {} captures, matching stream gain ranges and clamped gain\n", captures);
}

void SweepRestarts(const Device::Info& info) {
    auto sweep = OpenSweep(info);
    const auto base = Settings(sweep->capabilities());
    const F64 lowRate = Rates(sweep->capabilities()).front();
    Sweep::Settings settings{
        .sampleRate = lowRate,
        .antenna = base.antenna,
        .automaticGain = false,
    };
    for (const auto rate : {lowRate, base.sampleRate}) {
        settings.sampleRate = rate;
        Success(sweep->configure(settings), "configure sweep restart rate");
        const Sweep::Request request{
            .frequencies = {base.frequency, base.frequency + lowRate},
            .sampleRate = sweep->applied().sampleRate,
            .captureLength = 2048,
        };
        for (U64 restart = 0; restart < 3; ++restart) {
            Success(sweep->start(request), "start/restart sweep");
            U64 captures = 0;
            const auto deadline = Clock::now() + 10s;
            Success(sweep->run([&] { return Clock::now() < deadline; }, [&](const auto& capture) {
                Require(capture.hop == captures % request.frequencies.size(), "restart hop mismatch");
                Require(capture.samples.size() == request.captureLength, "short restart capture");
                return ++captures < 6;
            }), "capture after sweep restart");
            Require(captures == 6, "sweep restart stalled");
            jst::fmt::print("  sweep restart {} at {:.3f} MS/s: {} captures\n",
                            restart + 1, rate / 1e6, captures);
        }
    }
}

void Recovery(const Device::Info& info) {
    for (U64 cycle = 0; cycle < 3; ++cycle) {
        {
            auto stream = OpenStream(info);
            const auto settings = Settings(stream->capabilities());
            Success(stream->start(settings), "start reopened stream");
            Applied(*stream, settings);
            Receive(*stream, 1s);
            Success(stream->start(settings), "restart reopened stream");
            Receive(*stream, 1s);
        }
        {
            auto sweep = OpenSweep(info);
            const auto base = Settings(sweep->capabilities());
            const Sweep::Settings settings{
                .sampleRate = base.sampleRate,
                .antenna = base.antenna,
                .automaticGain = false,
            };
            Success(sweep->configure(settings), "configure reopened sweep");
            const Sweep::Request request{
                .frequencies = {base.frequency},
                .sampleRate = sweep->applied().sampleRate,
                .captureLength = 2048,
            };
            for (U64 restart = 0; restart < 2; ++restart) {
                Success(sweep->start(request), "start/restart reopened sweep");
                bool captured = false;
                const auto deadline = Clock::now() + 5s;
                Success(sweep->run([&] { return Clock::now() < deadline; }, [&](const auto& capture) {
                    Require(capture.samples.size() == request.captureLength, "short recovery capture");
                    captured = true;
                    return false;
                }), "run reopened sweep");
                Require(captured, "reopened sweep stalled");
            }
        }
        jst::fmt::print("  close/reopen cycle {} passed\n", cycle + 1);
    }
}

}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    JST_LOG_SET_DEBUG_LEVEL(2);
    JST_LOG_COLOR(false);
    jst::fmt::print("SDR hardware checks: +/-30% throughput, <=2 gaps/s, <=1% reported loss.\n"
                    "Loss counters and applied readbacks have driver-specific coverage; this is not "
                    "an independent RF freshness or lossless-FFT proof.\n");
    Results results;
    std::vector<Device::Info> devices;
    results.run("discovery", [&] { devices = Device::List(true); });
    for (const auto& driver : Device::Drivers()) {
        if (std::none_of(devices.begin(), devices.end(), [&](const auto& info) {
            return info.driver == driver.name;
        })) {
            jst::fmt::print("[SKIP] {}: no connected device\n", driver.name);
        }
    }
    if (devices.empty()) {
        jst::fmt::print("[SKIP] No SDR hardware found\n");
        return results.failed ? 1 : 77;
    }
    std::set<std::string> selectors;
    for (const auto& info : devices) {
        jst::fmt::print("\nDEVICE {} [{}]\n", info.label, info.selector());
        std::vector<F64> rates;
        results.run("enumeration and capabilities", [&] {
            Require(!info.label.empty() && !info.serial.empty() && !info.driver.empty(),
                    "device label, serial or driver is missing");
            Require(selectors.insert(info.selector()).second, "duplicate device selector");
            {
                auto stream = OpenStream(info);
                rates = Rates(stream->capabilities());
                const auto settings = Settings(stream->capabilities());
                Success(stream->start(settings), "start capability check");
                Applied(*stream, settings);
            }
            auto sweep = OpenSweep(info);
            Capabilities(sweep->capabilities());
            const auto& firmware = sweep->capabilities().firmwareSweep;
            Require(!firmware || info.driver == "hackrf", "unexpected firmware sweep driver");
            if (firmware) {
                Require(firmware->layout && firmware->maxDwell > 0 && firmware->captureStep > 0 &&
                        firmware->maxCaptureLength >= firmware->captureBase &&
                        Device::Range::Contains(sweep->capabilities().sampleRateRanges,
                                                firmware->sampleRate), "invalid firmware capabilities");
            }
        });
        for (const auto rate : rates) {
            results.run(jst::fmt::format("throughput {:.3f} MS/s", rate / 1e6),
                        [&] { Throughput(info, rate); });
        }
        results.run("streaming controls and restart", [&] { Controls(info); });
        results.run("retune sweep and restart", [&] { Sweeps(info, Plan::Mode::Retune); });
        results.run("sweep/stream cross-band gain consistency", [&] { SweepGains(info); });
        results.run("low-rate sweep restart and rate recovery", [&] { SweepRestarts(info); });
        if (info.driver == "hackrf") {
            results.run("firmware sweep and restart", [&] { Sweeps(info, Plan::Mode::Firmware); });
        }
        results.run("stream/sweep close and reopen", [&] { Recovery(info); });
    }
    jst::fmt::print("\nHardware results: {} passed, {} failed, {} devices\n",
                    results.passed, results.failed, devices.size());
    return results.failed ? 1 : 0;
}
