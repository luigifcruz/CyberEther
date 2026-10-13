#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include "domains/io/sdr/tools/samples.hh"
#include "domains/io/sdr/devices/hackrf/sweep.hh"
#include "domains/io/sdr/devices/hackrf/sweep_decoder.hh"
#include "domains/io/sdr/devices/hackrf/transfer_queue.hh"
#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/multiply/module.hh"
#include "jetstream/domains/dsp/amplitude/module.hh"
#include "jetstream/domains/dsp/fft/module.hh"
#include "jetstream/domains/dsp/invert/module.hh"
#include "jetstream/domains/dsp/window/module.hh"
#include "jetstream/domains/io/sdr_sweep/block.hh"
#include "jetstream/domains/io/sdr_sweep/module.hh"
#include "jetstream/domains/visualization/spectrum_analyzer/block.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/testing.hh"
#include "module_impl.hh"
#include "scripted_radio.hh"
#include "domains/io/sdr_sweep/tools/sweep.hh"

using namespace Jetstream;
using namespace TestRadio;
using Catch::Matchers::WithinAbs;

namespace {

using Mode = Modules::Sweep::Plan::Mode;

Radio::Device::Capabilities TestCaps(const F64 dcExclusion = 0.0) {
    Radio::Device::Capabilities caps;
    caps.frequencyRanges = {{10.0e6, 2000.0e6}};
    caps.sampleRateRanges = {{1.0e6, 10.0e6}};
    caps.sweepSampleRate = 2.0e6;
    caps.usableBandwidth = 0.75;
    caps.dcExclusion = dcExclusion;
    return caps;
}

Radio::Device::Capabilities HackRfCaps() {
    auto caps = TestCaps();
    caps.sampleRateRanges = {{2.0e6, 20.0e6}};
    caps.sweepSampleRate = 20.0e6;
    caps.firmwareSweep = Radio::HackRfSweep::Capability();
    return caps;
}

Modules::Sweep::Plan RequirePlan(const Modules::Sweep::Plan::Request& request,
                                 const Radio::Device::Capabilities& caps) {
    Modules::Sweep::Plan plan;
    REQUIRE(Modules::Sweep::Plan::Create(request, caps, plan) == Result::SUCCESS);
    return plan;
}

F64 BinCenter(const Modules::Sweep::Plan& plan, const F64 bin) {
    return plan.startFrequency + (bin + 0.5) * plan.binWidth;
}

// Captures of one hop observing tones through the hop LO. Each capture of a
// dwell starts at `start` samples of a continuous clock. Tones outside the
// capture's Nyquist band are removed, as the front-end filter would.
std::vector<CF32> HopCapture(const Modules::Sweep::Plan& plan, const U64 hop,
                             const std::vector<Tone>& tones, const U64 start = 0,
                             const std::vector<CF32>& noise = {}) {
    std::vector<CF32> samples(plan.captureSamples());
    const F64 lo = plan.hops[hop].frequency;
    for (U64 n = 0; n < samples.size(); ++n) {
        const F64 time = static_cast<F64>(start + n) / plan.sampleRate;
        std::complex<F64> value;
        for (const auto& tone : tones) {
            if (std::abs(tone.frequency - lo) >= plan.sampleRate / 2.0) {
                continue;
            }
            value += std::polar(tone.amplitude, 2.0 * std::numbers::pi * (tone.frequency - lo) * time);
        }
        samples[n] = CF32{static_cast<F32>(value.real()), static_cast<F32>(value.imag())};
        if (!noise.empty()) {
            samples[n] += noise[(hop * samples.size() + n) % noise.size()];
        }
    }
    return samples;
}

std::vector<CF32> Synthesize(const Modules::Sweep::Plan& plan, const std::vector<Tone>& tones,
                             const std::vector<CF32>& noise = {}) {
    Modules::Sweep::Assembler assembler;
    REQUIRE(assembler.configure(plan) == Result::SUCCESS);
    for (U64 hop = 0; hop < plan.hops.size(); ++hop) {
        const auto samples = HopCapture(plan, hop, tones, 0, noise);
        const Radio::Device::SweepCapture capture{
            .hop = hop,
            .frequency = plan.hops[hop].frequency,
            .samples = samples,
        };
        REQUIRE(assembler.add(capture) == Result::SUCCESS);
    }
    REQUIRE(assembler.complete());
    std::vector<CF32> frame(plan.outputBins);
    REQUIRE(assembler.synthesize(frame) == Result::SUCCESS);
    return frame;
}

// Runs a frame through the modules of the Spectrum Analyzer chain: Blackman
// window, Invert, Multiply, FFT, and Amplitude. Returns dBFS per display bin.
std::vector<F32> SpectrumAnalyzerDb(const std::vector<CF32>& frame) {
    const U64 size = frame.size();
    const auto run = [](TestContext& context) {
        REQUIRE(context.run() == Result::SUCCESS);
    };
    const auto axes = [](Tensor& tensor) {
        REQUIRE(SetSignalAxes(tensor, {.sample = Index{0}}) == Result::SUCCESS);
    };

    Tensor input(DeviceType::CPU, DataType::CF32, {size});
    std::copy(frame.begin(), frame.end(), input.data<CF32>());
    axes(input);

    TestContext window("window", DeviceType::CPU, RuntimeType::NATIVE, "generic");
    Modules::Window windowConfig;
    windowConfig.size = size;
    window.setConfig(windowConfig);
    run(window);
    Tensor taps = window.output("window");
    axes(taps);

    TestContext invert("invert", DeviceType::CPU, RuntimeType::NATIVE, "generic");
    invert.setConfig(Modules::Invert{});
    invert.setInput("signal", taps);
    run(invert);
    Tensor inverted = invert.output("signal");
    axes(inverted);

    TestContext multiply("multiply", DeviceType::CPU, RuntimeType::NATIVE, "generic");
    multiply.setConfig(Modules::Multiply{});
    multiply.setInput("a", input);
    multiply.setInput("b", inverted);
    run(multiply);
    Tensor product = multiply.output("product");
    axes(product);

    TestContext fft("fft", DeviceType::CPU, RuntimeType::NATIVE, "generic");
    Modules::Fft fftConfig;
    fftConfig.forward = true;
    fft.setConfig(fftConfig);
    fft.setInput("signal", product);
    run(fft);
    Tensor spectrum = fft.output("signal");
    axes(spectrum);

    TestContext amplitude("amplitude", DeviceType::CPU, RuntimeType::NATIVE, "generic");
    amplitude.setConfig(Modules::Amplitude{});
    amplitude.setInput("signal", spectrum);
    run(amplitude);
    const auto& decibels = amplitude.output("signal");
    return {decibels.data<F32>(), decibels.data<F32>() + size};
}

// A directly captured tone of the same amplitude at display bin `bin`.
std::vector<CF32> DirectTone(const U64 size, const F64 bin, const F64 amplitude) {
    std::vector<CF32> frame(size);
    const F64 offset = bin - static_cast<F64>(size / 2);
    for (U64 n = 0; n < size; ++n) {
        const auto value = std::polar(amplitude, 2.0 * std::numbers::pi * offset *
                                                     static_cast<F64>(n) / static_cast<F64>(size));
        frame[n] = CF32{static_cast<F32>(value.real()), static_cast<F32>(value.imag())};
    }
    return frame;
}

F32 PeakBinLevel(const std::vector<F32>& decibels, const U64 bin) {
    return decibels[bin];
}

U64 ArgMax(const std::vector<F32>& values) {
    return static_cast<U64>(std::max_element(values.begin(), values.end()) - values.begin());
}

std::vector<CF32> WhiteNoise(const U64 count, const F64 variance, const U32 seed) {
    std::mt19937 generator(seed);
    const F64 sigma = std::sqrt(variance / 2.0);
    std::vector<CF32> noise(count);
    for (auto& sample : noise) {
        const F64 u1 = (static_cast<F64>(generator()) + 1.0) / 4294967297.0;
        const F64 u2 = static_cast<F64>(generator()) / 4294967296.0;
        const F64 radius = sigma * std::sqrt(-2.0 * std::log(u1));
        const F64 angle = 2.0 * std::numbers::pi * u2;
        sample = CF32{static_cast<F32>(radius * std::cos(angle)),
                      static_cast<F32>(radius * std::sin(angle))};
    }
    return noise;
}

}  // namespace

//
// Planning.
//

TEST_CASE("SDR Sweep plans a uniform half-open grid with one contributor per bin",
          "[modules][sdr_sweep][plan]") {
    F64 resolution = 0.0;
    F64 dc = 0.0;
    SECTION("even capture length") {
        resolution = 10.0e3;
    }
    SECTION("odd capture length") {
        resolution = 9.0e3;
    }
    SECTION("DC exclusion") {
        resolution = 10.0e3;
        dc = 25.0e3;
    }

    const auto plan = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 120.0e6,
                                   .resolution = resolution}, TestCaps(dc));
    const U64 length = static_cast<U64>(std::ceil(2.0e6 / resolution));
    REQUIRE(plan.mode == Mode::Retune);
    REQUIRE(plan.segmentLength == length);
    REQUIRE(plan.binWidth == 2.0e6 / static_cast<F64>(length));
    REQUIRE(plan.binWidth <= resolution);
    REQUIRE(plan.startFrequency == 100.0e6);
    REQUIRE(plan.endFrequency() >= 120.0e6);
    REQUIRE(plan.endFrequency() < 120.0e6 + plan.binWidth);
    REQUIRE(plan.outputBins == static_cast<U64>(std::ceil(20.0e6 / plan.binWidth)));
    REQUIRE(plan.span() == static_cast<F64>(plan.outputBins) * plan.binWidth);
    REQUIRE(plan.centerFrequency() == BinCenter(plan, static_cast<F64>(plan.outputBins / 2)));

    const auto center = static_cast<I64>(length / 2);
    const auto half = static_cast<I64>(std::floor(0.75 * static_cast<F64>(length) / 2.0));
    const auto dcBins = static_cast<I64>(std::ceil(dc / plan.binWidth));
    std::vector<U64> contributors(plan.outputBins, 0);
    for (const auto& hop : plan.hops) {
        for (const auto& run : hop.runs) {
            for (U64 i = 0; i < run.length; ++i) {
                const auto offset = static_cast<I64>(run.segment + i) - center;
                REQUIRE(offset >= -half);
                REQUIRE(offset < half);
                REQUIRE(std::abs(offset) >= dcBins);
                ++contributors[run.grid + i];
                // Hops tune on whole hertz, so a bin lands within half a hertz of its grid frequency.
                const F64 segmentFrequency = hop.frequency + static_cast<F64>(offset) * plan.binWidth;
                REQUIRE_THAT(segmentFrequency,
                             WithinAbs(BinCenter(plan, static_cast<F64>(run.grid + i)), 0.5));
            }
        }
    }
    REQUIRE(std::all_of(contributors.begin(), contributors.end(),
                        [](const U64 count) { return count == 1; }));
}

TEST_CASE("SDR Sweep places retune hops on whole hertz so fine bins survive driver rounding",
          "[modules][sdr_sweep][plan][rtlsdr]") {
    auto caps = TestCaps(5.0e3);
    caps.sampleRateRanges = {{225001.0, 300000.0}, {900001.0, 3200000.0}};
    const auto plan = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 100.1e6,
                                   .sampleRate = 240.0e3, .resolution = 1.0}, caps);
    REQUIRE(plan.mode == Mode::Retune);
    REQUIRE(plan.binWidth < 10.0);
    REQUIRE(plan.hops.size() > 1);

    Modules::Sweep::Assembler assembler;
    REQUIRE(assembler.configure(plan) == Result::SUCCESS);
    const std::vector<CF32> samples(plan.captureSamples());
    for (U64 hop = 0; hop < plan.hops.size(); ++hop) {
        const F64 frequency = plan.hops[hop].frequency;
        REQUIRE(frequency == std::round(frequency));
        REQUIRE(assembler.add({.hop = hop, .frequency = static_cast<F64>(std::llround(frequency)),
                               .samples = samples}) == Result::SUCCESS);
    }
    REQUIRE(assembler.complete());
}

TEST_CASE("SDR Sweep rejects invalid and impractical plans with guidance",
          "[modules][sdr_sweep][plan][limits]") {
    const Modules::Sweep::Plan::Request valid{
        .startFrequency = 100.0e6, .endFrequency = 120.0e6, .resolution = 10.0e3,
    };
    auto caps = TestCaps();
    auto request = valid;
    Modules::Sweep::Plan plan;

    SECTION("start at or above end") {
        request.endFrequency = request.startFrequency;
    }
    SECTION("non-finite frequency") {
        request.startFrequency = std::numeric_limits<F64>::quiet_NaN();
    }
    SECTION("resolution not positive") {
        request.resolution = 0.0;
    }
    SECTION("dwell outside limits") {
        request.dwell = Modules::Sweep::MaxDwell + 1;
    }
    SECTION("capture rate the radio does not support") {
        request.sampleRate = 20.0e6;
    }
    SECTION("span cannot fit even the smallest capture") {
        request.endFrequency = 1e300;
    }
    SECTION("hops outside the tunable range") {
        request.startFrequency = 5.0e6;
    }
    SECTION("explicit firmware mode the radio lacks") {
        request.mode = Mode::Firmware;
    }
    REQUIRE(Modules::Sweep::Plan::Create(request, caps, plan) == Result::ERROR);
    REQUIRE(Modules::Sweep::Plan::Create(valid, TestCaps(), plan) == Result::SUCCESS);
}

TEST_CASE("SDR Sweep coarsens only enough to fit the final output grid",
          "[modules][sdr_sweep][plan][adaptive]") {
    using namespace Modules::Sweep;
    auto caps = TestCaps();
    caps.frequencyRanges = {{10.0e6, 6000.0e6}};
    Plan::Request request{.startFrequency = 100.0e6, .endFrequency = 3012.0e6,
                     .resolution = 10.0e3};
    const auto wide = RequirePlan(request, caps);
    REQUIRE((request.endFrequency - request.startFrequency) / request.resolution == 291200);
    REQUIRE(wide.segmentLength == 180);
    REQUIRE(wide.outputBins == 262080);
    REQUIRE(wide.outputBins <= MaxOutputBins);
    REQUIRE(wide.workingSetBytes() <= MaxWorkingSetBytes);
    REQUIRE(wide.endFrequency() >= request.endFrequency);
    REQUIRE(wide.binWidth > request.resolution);
    REQUIRE(std::ceil((request.endFrequency - request.startFrequency) /
                       (wide.sampleRate / static_cast<F64>(wide.segmentLength + 1))) > MaxOutputBins);

    request.endFrequency = 120.0e6;
    const auto narrow = RequirePlan(request, caps);
    REQUIRE(narrow.segmentLength == 200);
    REQUIRE(narrow.binWidth == request.resolution);

    request.resolution = 5.0e3;
    request.endFrequency = request.startFrequency + MaxOutputBins * request.resolution;
    const auto exact = RequirePlan(request, caps);
    REQUIRE(exact.outputBins == MaxOutputBins);
    REQUIRE(exact.segmentLength == 400);
    request.endFrequency += 1.0;
    const auto over = RequirePlan(request, caps);
    REQUIRE(over.segmentLength == 399);
    REQUIRE(over.outputBins <= MaxOutputBins);
    REQUIRE(over.endFrequency() >= request.endFrequency);

    request.endFrequency = request.startFrequency + 50.0e3;
    request.resolution = 10.0e3;
    const auto small = RequirePlan(request, caps);
    REQUIRE(small.outputBins == MinOutputBins);
    REQUIRE(small.segmentLength == 601);
    request.endFrequency = 120.0e6;
    request.resolution = 200.0e3;
    const auto shortCapture = RequirePlan(request, caps);
    REQUIRE(shortCapture.segmentLength == MinSegmentLength);
}

TEST_CASE("SDR Sweep bounds capture lengths and firmware padding before allocation",
          "[modules][sdr_sweep][plan][adaptive][hackrf]") {
    using namespace Modules::Sweep;
    const auto capture = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 101.0e6,
                                      .resolution = 10.0, .dwell = MaxDwell}, TestCaps());
    REQUIRE(capture.segmentLength == MaxSegmentLength);
    REQUIRE(capture.workingSetBytes() <= MaxWorkingSetBytes);

    auto caps = HackRfCaps();
    caps.frequencyRanges = {{0.0, 6000.0e6}};
    const auto padded = RequirePlan({.startFrequency = 88.3e6, .endFrequency = 2707.9e6,
                                     .resolution = 10.0e3, .mode = Mode::Firmware}, caps);
    REQUIRE(padded.mode == Mode::Firmware);
    REQUIRE(padded.startFrequency == 88.0e6);
    REQUIRE(padded.segmentLength == 1996);
    REQUIRE((padded.segmentLength + 4) % 8 == 0);
    REQUIRE(padded.outputBins == 261476);
    REQUIRE(padded.endFrequency() >= 2707.9e6);
    REQUIRE(131 * (padded.segmentLength + 8) > MaxOutputBins);

    // The capture length never exceeds what the device advertises.
    const auto payload = RequirePlan({.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                                      .resolution = 1.0}, caps);
    REQUIRE(payload.segmentLength == caps.firmwareSweep->maxCaptureLength);
    REQUIRE(payload.mode == Mode::Firmware);

    // Auto can use a valid retune plan when firmware endpoint padding cannot
    // fit. An explicitly selected firmware method still reports that limit.
    Plan::Request edge{.startFrequency = 5990.0e6, .endFrequency = 5991.0e6,
                  .resolution = 100.0e3};
    REQUIRE(RequirePlan(edge, caps).mode == Mode::Retune);
    edge.mode = Mode::Firmware;
    Plan rejected;
    REQUIRE(Plan::Create(edge, caps, rejected) == Result::ERROR);
}

TEST_CASE("SDR Sweep Auto preserves targets that Retune can meet without firmware coarsening",
          "[modules][sdr_sweep][plan][adaptive][hackrf]") {
    using namespace Modules::Sweep;
    auto caps = HackRfCaps();
    Plan::Request request{.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                     .sampleRate = 20.0e6, .resolution = 1.0e3};
    const auto native = RequirePlan(request, caps);
    REQUIRE(native.mode == Mode::Retune);
    REQUIRE(native.segmentLength == 20000);
    REQUIRE(native.outputBins == 20000);
    REQUIRE(native.binWidth == request.resolution);

    request.mode = Mode::Firmware;
    const auto explicitFirmware = RequirePlan(request, caps);
    REQUIRE(explicitFirmware.mode == Mode::Firmware);
    REQUIRE(explicitFirmware.segmentLength == 8180);
    REQUIRE(explicitFirmware.binWidth > request.resolution);

    request.mode = Mode::Automatic;
    request.resolution = 10.0e3;
    REQUIRE(RequirePlan(request, caps).mode == Mode::Firmware);

    // A supported method still needs a physically valid plan. Candidate errors
    // must not prevent using the valid firmware sequence.
    request.resolution = 1.0e3;
    caps.frequencyRanges = {{0.0, 108.0e6}};
    REQUIRE(RequirePlan(request, caps).mode == Mode::Firmware);
}

TEST_CASE("SDR Sweep Auto retains bounded firmware adaptation when neither method meets the target",
          "[modules][sdr_sweep][plan][adaptive][hackrf]") {
    using namespace Modules::Sweep;
    auto caps = HackRfCaps();
    caps.frequencyRanges = {{0.0, 6000.0e6}};
    Plan::Request request{.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                     .sampleRate = 20.0e6, .resolution = 1.0};
    SECTION("capture-length limits") {}
    SECTION("final output-bin limits") {
        request.endFrequency = request.startFrequency + 2912.0e6;
        request.resolution = 1.0e3;
    }
    const auto firmware = RequirePlan(request, caps);
    REQUIRE(firmware.mode == Mode::Firmware);
    REQUIRE(firmware.binWidth > request.resolution);
    REQUIRE(firmware.outputBins <= MaxOutputBins);
    REQUIRE(firmware.workingSetBytes() <= MaxWorkingSetBytes);
    REQUIRE(firmware.endFrequency() >= request.endFrequency);
    request.mode = Mode::Retune;
    const auto retune = RequirePlan(request, caps);
    REQUIRE(retune.binWidth > request.resolution);
    REQUIRE(retune.outputBins <= MaxOutputBins);
    REQUIRE(retune.workingSetBytes() <= MaxWorkingSetBytes);
    REQUIRE(retune.endFrequency() >= request.endFrequency);
}

TEST_CASE("SDR Sweep respects fixed firmware timing and explicit methods",
          "[modules][sdr_sweep][plan][hackrf][settling]") {
    auto caps = HackRfCaps();
    Modules::Sweep::Plan::Request request{.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                                    .resolution = 10.0e3};
    REQUIRE(RequirePlan(request, caps).mode == Mode::Firmware);
    request.settlingTime = 0.010;
    REQUIRE(RequirePlan(request, caps).mode == Mode::Retune);
    request.mode = Mode::Firmware;
    Modules::Sweep::Plan plan;
    REQUIRE(Modules::Sweep::Plan::Create(request, caps, plan) == Result::ERROR);
    request.settlingTime = 0.0;
    REQUIRE(RequirePlan(request, caps).mode == Mode::Firmware);

    request.mode = Mode::Retune;
    request.sampleRate = 10.0e6;
    REQUIRE(RequirePlan(request, caps).sampleRate == 10.0e6);
    REQUIRE(RequirePlan(request, caps).mode == Mode::Retune);
}

TEST_CASE("SDR Sweep picks generic HackRF retuning only when firmware cannot honor the settings",
          "[modules][sdr_sweep][plan][hackrf][browser]") {
    // The same capabilities come from native USB and WebUSB, so these rules
    // decide the method in the browser as well.
    const auto caps = HackRfCaps();
    const Modules::Sweep::Plan::Request defaults{.startFrequency = 88.0e6,
                                                 .endFrequency = 108.0e6,
                                                 .resolution = 10.0e3};
    REQUIRE(RequirePlan(defaults, caps).mode == Mode::Firmware);
    auto request = defaults;
    request.sampleRate = 20.0e6;
    REQUIRE(RequirePlan(request, caps).mode == Mode::Firmware);

    SECTION("another capture rate") {
        request.sampleRate = 8.0e6;
    }
    SECTION("a settling override") {
        request.settlingTime = 0.005;
    }
    SECTION("a resolution only retuning meets") {
        request.resolution = 1.0e3;
    }
    const auto plan = RequirePlan(request, caps);
    REQUIRE(plan.mode == Mode::Retune);
    REQUIRE(plan.binWidth <= request.resolution);
    REQUIRE(plan.deviceRequest().firmware == false);
    REQUIRE(plan.deviceRequest().frequencies.size() == plan.hops.size());

    // Real device limits still reject impossible requests.
    Modules::Sweep::Plan rejected;
    request.sampleRate = 25.0e6;
    REQUIRE(Modules::Sweep::Plan::Create(request, caps, rejected) == Result::ERROR);
}

TEST_CASE("SDR Sweep limits keep the source and default display within budget",
          "[modules][sdr_sweep][plan][limits]") {
    using namespace Modules::Sweep;
    // Worst-case working set allowed by the plan limits.
    Plan worst;
    worst.segmentLength = MaxSegmentLength;
    worst.dwell = MaxDwell;
    worst.outputBins = MaxOutputBins;
    REQUIRE(worst.workingSetBytes() <= MaxWorkingSetBytes);

    // Spectrum Analyzer waterfalls hold four bytes per bin and row, indexed
    // with I32. The documented height for the largest output fits 256 MiB.
    REQUIRE(MaxOutputBins * 1024 <= static_cast<U64>(std::numeric_limits<I32>::max()));
    REQUIRE(MaxOutputBins * 256 * sizeof(F32) == U64{256} << 20);

    // The default block configuration stays small with the default display.
    const auto plan = RequirePlan({.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                                   .resolution = 10.0e3}, TestCaps());
    REQUIRE(plan.outputBins == 2000);
    REQUIRE(plan.outputBins * 1024 * sizeof(F32) < U64{16} << 20);
}

TEST_CASE("SDR Sweep firmware sweeps place HackRF bands on the output grid",
          "[modules][sdr_sweep][plan][hackrf]") {
    const auto caps = HackRfCaps();

    const auto plan = RequirePlan({.startFrequency = 88.3e6, .endFrequency = 107.9e6,
                                   .resolution = 10.0e3}, caps);
    REQUIRE(plan.mode == Mode::Firmware);
    REQUIRE(plan.segmentLength == 2004);
    REQUIRE((plan.segmentLength + 4) % 8 == 0);
    REQUIRE(plan.sampleRate == 20.0e6);
    REQUIRE(plan.startFrequency == 88.0e6);
    REQUIRE(plan.outputBins == 2004);
    REQUIRE(plan.hops.size() == 2);
    REQUIRE(plan.hops[0].frequency == 95.5e6);
    REQUIRE(plan.hops[1].frequency == 100.5e6);

    // `hackrf_sweep` places the upper quarter band of header frequency `f` at
    // `[f, f + 5 MHz)` and the lower one at `[f + 10 MHz, f + 15 MHz)`.
    const U64 quarter = plan.segmentLength / 4;
    for (const auto& hop : plan.hops) {
        const F64 header = hop.frequency - Radio::HackRfSweep::Offset;
        const auto first = static_cast<U64>(std::llround((header - plan.startFrequency) / plan.binWidth));
        REQUIRE(hop.runs.size() == 2);
        REQUIRE(hop.runs[0] == Modules::Sweep::Plan::Run{
            .segment = (1 + plan.segmentLength * 5 / 8 + plan.segmentLength / 2) % plan.segmentLength,
            .grid = first, .length = quarter});
        REQUIRE(hop.runs[1].grid == first + plan.segmentLength / 2);
        REQUIRE(hop.runs[1].length == quarter);
    }

    // Tones land on their grid bins through the firmware mapping as well.
    const Tone tone{.frequency = BinCenter(plan, 777), .amplitude = 0.25};
    const auto frame = Synthesize(plan, {tone});
    const auto decibels = SpectrumAnalyzerDb(frame);
    REQUIRE(ArgMax(decibels) == 777);

    SECTION("automatic selection falls back when firmware limits do not fit") {
        REQUIRE(Modules::Sweep::Plan::ResolveMode({.resolution = 10.0e3}, caps) ==
                Mode::Firmware);
        REQUIRE(Modules::Sweep::Plan::ResolveMode({.sampleRate = 10.0e6, .resolution = 10.0e3}, caps) ==
                Mode::Retune);
        REQUIRE(Modules::Sweep::Plan::ResolveMode({.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                                               .resolution = 1.0e3}, caps) == Mode::Retune);
        Modules::Sweep::Plan rejected;
        REQUIRE(Modules::Sweep::Plan::Create({.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                                           .sampleRate = 10.0e6, .resolution = 10.0e3,
                                           .mode = Mode::Firmware},
                                          caps, rejected) == Result::ERROR);
    }
}

//
// Synthesis and display normalization.
//

TEST_CASE("SDR Sweep synthetic IQ shows tones at the direct Spectrum Analyzer level",
          "[modules][sdr_sweep][dsp][normalization]") {
    F64 resolution = 10.0e3;
    F64 dc = 0.0;
    F64 end = 104.0e6;
    SECTION("even capture and output lengths") {}
    SECTION("odd capture length") {
        resolution = 9.0e3;
    }
    SECTION("odd output length") {
        end = 104.005e6;
    }
    SECTION("DC-excluded captures") {
        dc = 25.0e3;
    }

    const auto plan = RequirePlan({.startFrequency = 100.0e6, .endFrequency = end,
                                   .resolution = resolution}, TestCaps(dc));
    const U64 bins = plan.outputBins;
    const F64 amplitude = 0.1;

    // Interior, seam, first, and last bins. The seam is the first bin of the
    // second hop's lower run.
    const auto& seamRun = plan.hops[1].runs.front();
    for (const U64 bin : {bins / 3, seamRun.grid, seamRun.grid - 1, U64{0}, bins - 1}) {
        DYNAMIC_SECTION("Bin " << bin << " of " << bins) {
            const auto decibels = SpectrumAnalyzerDb(
                Synthesize(plan, {{.frequency = BinCenter(plan, static_cast<F64>(bin)),
                                   .amplitude = amplitude}}));
            const auto reference = SpectrumAnalyzerDb(DirectTone(bins, static_cast<F64>(bin), amplitude));
            REQUIRE(ArgMax(decibels) == bin);
            REQUIRE_THAT(PeakBinLevel(decibels, bin), WithinAbs(reference[bin], 0.05));

            // Display bin `k` is labeled `center + (k - floor(M / 2)) * span / M`.
            const F64 label = plan.centerFrequency() +
                              (static_cast<F64>(bin) - static_cast<F64>(bins / 2)) * plan.span() /
                              static_cast<F64>(bins);
            REQUIRE_THAT(label, WithinAbs(BinCenter(plan, static_cast<F64>(bin)), 1e-6));

            // The display transform is circular, so edge tones leak across
            // the opposite edge exactly like a direct capture does.
            if (bin == 0) {
                REQUIRE_THAT(decibels[bins - 1], WithinAbs(decibels[1], 0.05));
                REQUIRE_THAT(decibels[bins - 1], WithinAbs(reference[bins - 1], 0.05));
            }
        }
    }
}

TEST_CASE("SDR Sweep characterizes off-bin tones, seam ripple, and spreading",
          "[modules][sdr_sweep][dsp][seams]") {
    const auto plan = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 104.0e6,
                                   .resolution = 10.0e3}, TestCaps());
    const U64 bins = plan.outputBins;
    const U64 seam = plan.hops[1].runs.front().grid;
    const F64 amplitude = 0.1;

    // Off-bin tones follow the Blackman scalloping of a direct capture.
    for (const F64 fraction : {0.25, 0.5}) {
        const F64 bin = static_cast<F64>(bins / 3) + fraction;
        const auto decibels = SpectrumAnalyzerDb(
            Synthesize(plan, {{.frequency = BinCenter(plan, bin), .amplitude = amplitude}}));
        const auto reference = SpectrumAnalyzerDb(DirectTone(bins, bin, amplitude));
        const auto peak = ArgMax(reference);
        REQUIRE(ArgMax(decibels) == peak);
        REQUIRE_THAT(decibels[peak], WithinAbs(reference[peak], 0.3));
    }

    // A tone crossing a seam keeps its level within 1 dB, and spurs outside
    // the main lobe stay 40 dB down. Each hop's captures are acquired at
    // different times, so this bounds the stitching artifact only.
    const auto interior = SpectrumAnalyzerDb(DirectTone(bins, static_cast<F64>(bins / 3) + 0.5, amplitude));
    const F32 scalloped = *std::max_element(interior.begin(), interior.end());
    for (F64 offset = -2.0; offset <= 2.0; offset += 0.25) {
        INFO("Offset " << offset);
        const F64 bin = static_cast<F64>(seam) + offset;
        const auto decibels = SpectrumAnalyzerDb(
            Synthesize(plan, {{.frequency = BinCenter(plan, bin), .amplitude = amplitude}}));
        const auto peak = ArgMax(decibels);
        REQUIRE(std::abs(static_cast<F64>(peak) - bin) <= 1.0);
        REQUIRE(decibels[peak] >= scalloped - 1.0f);
        F32 spur = -std::numeric_limits<F32>::infinity();
        for (U64 k = 0; k < bins; ++k) {
            if (std::abs(static_cast<F64>(k) - bin) > 4.0) {
                spur = std::max(spur, decibels[k]);
            }
        }
        REQUIRE(spur <= decibels[peak] - 40.0f);
    }
}

TEST_CASE("SDR Sweep noise floor follows the capture bin width",
          "[modules][sdr_sweep][dsp][noise]") {
    const auto plan = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 104.0e6,
                                   .resolution = 10.0e3}, TestCaps());
    const F64 variance = 1.0e-4;
    constexpr U32 Realizations = 8;

    // Each displayed bin averages independent capture bins weighted by the
    // display window, so its mean power is `sum(w^2) / M * variance / N`.
    // That equals the floor of a direct capture with the same bin width and
    // noise density, and is not rescaled per sweep.
    const U64 bins = plan.outputBins;
    F64 energy = 0.0;
    for (U64 n = 0; n < bins; ++n) {
        const F64 tap = 0.42 - 0.5 * std::cos(2.0 * std::numbers::pi * n / (bins - 1)) +
                        0.08 * std::cos(4.0 * std::numbers::pi * n / (bins - 1));
        energy += tap * tap;
    }
    const F64 expected = energy / static_cast<F64>(bins) * variance /
                         static_cast<F64>(plan.segmentLength);
    F64 measured = 0.0;
    for (U32 seed = 1; seed <= Realizations; ++seed) {
        const auto noise = WhiteNoise(plan.captureSamples() * plan.hops.size(), variance, seed);
        for (const auto value : SpectrumAnalyzerDb(Synthesize(plan, {}, noise))) {
            measured += std::pow(10.0, value / 10.0);
        }
    }
    measured /= static_cast<F64>(bins) * Realizations;
    REQUIRE_THAT(10.0 * std::log10(measured / expected), WithinAbs(0.0, 0.3));
}

TEST_CASE("SDR Sweep dwell averages magnitudes without cancelling phases",
          "[modules][sdr_sweep][dsp][dwell]") {
    auto plan = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 104.0e6,
                             .resolution = 10.0e3, .dwell = 3}, TestCaps());
    REQUIRE(plan.captureSamples() == 3 * plan.segmentLength);

    // A tone half a bin off the capture grid changes phase between captures.
    const U64 bin = plan.outputBins / 3;
    Modules::Sweep::Assembler assembler;
    REQUIRE(assembler.configure(plan) == Result::SUCCESS);
    const Tone tone{.frequency = BinCenter(plan, static_cast<F64>(bin)), .amplitude = 0.2};
    for (U64 hop = 0; hop < plan.hops.size(); ++hop) {
        auto samples = HopCapture(plan, hop, {tone});
        // Rotate the second capture so complex averaging would cancel it.
        for (U64 n = plan.segmentLength; n < 2 * plan.segmentLength; ++n) {
            samples[n] = -samples[n];
        }
        REQUIRE(assembler.add({.hop = hop, .frequency = plan.hops[hop].frequency,
                               .samples = samples}) == Result::SUCCESS);
    }
    REQUIRE(assembler.complete());
    REQUIRE_THAT(std::abs(assembler.spectrum()[bin]), WithinAbs(0.2, 1e-4));
}

TEST_CASE("SDR Sweep never completes partial, stale, or mistuned coverage",
          "[modules][sdr_sweep][dsp][sequence]") {
    const auto plan = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 104.0e6,
                                   .resolution = 10.0e3}, TestCaps());
    const U64 hops = plan.hops.size();
    REQUIRE(hops >= 3);
    Modules::Sweep::Assembler assembler;
    REQUIRE(assembler.configure(plan) == Result::SUCCESS);
    std::vector<CF32> frame(plan.outputBins);

    const Tone first{.frequency = BinCenter(plan, 10), .amplitude = 0.5};
    const auto add = [&](const U64 hop, const std::vector<Tone>& tones, const F64 detune = 0.0) {
        const auto samples = HopCapture(plan, hop, tones);
        return assembler.add({.hop = hop, .frequency = plan.hops[hop].frequency + detune,
                              .samples = samples});
    };

    // Hops before the first hop of a sweep are ignored.
    REQUIRE(add(1, {first}) == Result::INCOMPLETE);
    REQUIRE_FALSE(assembler.complete());
    REQUIRE(assembler.synthesize(frame) == Result::ERROR);

    // A missing hop invalidates the sweep, and assembly waits for hop zero.
    REQUIRE(add(0, {first}) == Result::SUCCESS);
    REQUIRE(add(2, {first}) == Result::INCOMPLETE);
    REQUIRE(assembler.completedHops() == 0);
    for (U64 hop = 1; hop < hops; ++hop) {
        REQUIRE(add(hop, {first}) == Result::INCOMPLETE);
    }
    REQUIRE_FALSE(assembler.complete());

    // A restarted sweep never inherits bins from the abandoned one.
    REQUIRE(add(0, {first}) == Result::SUCCESS);
    REQUIRE(add(1, {first}) == Result::SUCCESS);
    for (U64 hop = 0; hop < hops; ++hop) {
        REQUIRE(add(hop, {}) == Result::SUCCESS);
    }
    REQUIRE(assembler.complete());
    REQUIRE(assembler.synthesize(frame) == Result::SUCCESS);
    REQUIRE(std::all_of(frame.begin(), frame.end(), [](const CF32 value) {
        return std::abs(value) < 1e-5f;
    }));

    // Tuning within the documented tolerance maps normally, beyond it fails.
    REQUIRE(add(0, {}, 0.04 * plan.binWidth) == Result::SUCCESS);
    REQUIRE(add(1, {}, 0.06 * plan.binWidth) == Result::ERROR);
    REQUIRE_FALSE(assembler.complete());
}

//
// Module.
//

namespace {

struct SweepAccess : Modules::SdrSweepImpl {
    using Frame = Modules::SdrSweepImpl::Frame;
    using MetricState = Modules::SdrSweepImpl::MetricState;

    void initializeSlot(const Clock::time_point now) {
        running = true;
        metrics.reset(now);
        staging.assign(1, CF32{0.5f, 0.0f});
        slot.assign(1, CF32{});
        buffer = Tensor(DeviceType::CPU, DataType::CF32, {1, 1});
    }

    bool completeAt(const Clock::time_point now) {
        const bool published = publishFrame(now);
        // A single-hop frame can have a zero legacy host duration. Speed must
        // instead use completed frames and time since the worker started.
        sweepDuration.publish(0.0);
        return published;
    }

    Result consumeAt(const Clock::time_point now) {
        return emitFrame(now);
    }

    void fail() {
        errored = true;
    }

    static bool Expired(const Modules::SdrSweepImpl& impl, const Frame& frame,
                        const Clock::time_point now) {
        return (impl.*&SweepAccess::frameExpired)(frame, now);
    }

    static auto sweepDurationMember() {
        return &SweepAccess::sweepDuration;
    }
};

std::shared_ptr<Module> BuildSweepModule() {
    const auto implementations = Registry::ListAvailableModules("sdr_sweep");
    REQUIRE_FALSE(implementations.empty());
    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("sdr_sweep", implementation.device, implementation.runtime,
                                  implementation.provider, module) == Result::SUCCESS);
    return module;
}

Modules::SdrSweep ScriptedSweep() {
    Modules::SdrSweep config;
    config.deviceString = Selector;
    config.startFrequency = 100.0e6;
    config.endFrequency = 104.0e6;
    config.resolution = 10.0e3;
    return config;
}

struct ScriptedFlowgraph : ScriptedDrivers, FlowgraphFixture {};

struct SweepCleanup {
    std::shared_ptr<Module> module;

    ~SweepCleanup() {
        if (module->state() == Module::State::CREATED) {
            static_cast<void>(module->destroy());
        }
    }
};

}  // namespace

TEST_CASE("SDR Sweep slot occupancy integrates real occupied time and decays when empty",
          "[modules][sdr_sweep][metrics]") {
    using Clock = Modules::SdrSweepImpl::Clock;
    const auto start = Clock::time_point{};
    SweepAccess::MetricState metrics;
    metrics.reset(start);
    metrics.advance(start + std::chrono::milliseconds(500), false);
    REQUIRE(metrics.snapshot.bufferHealth == 0.0f);
    REQUIRE(metrics.snapshot.bufferLoss == 0.0);
    REQUIRE(metrics.snapshot.sweepSpeed == 0.0f);

    metrics.publish(start + std::chrono::milliseconds(500), false);
    metrics.advance(start + std::chrono::milliseconds(550), true);
    REQUIRE(metrics.health > 0.0);
    REQUIRE(metrics.snapshot.bufferHealth == 0.0f);
    metrics.advance(start + std::chrono::seconds(1), true);
    const F64 occupied = 1.0 - std::exp(-0.5 / 1.5);
    REQUIRE_THAT(metrics.snapshot.bufferHealth, WithinAbs(occupied, 1e-6));
    REQUIRE(metrics.snapshot.bufferHealth < 1.0f);

    // Consume at one second, then leave the actual slot empty for a second.
    metrics.advance(start + std::chrono::seconds(1), true);
    metrics.advance(start + std::chrono::seconds(2), false);
    REQUIRE_THAT(metrics.snapshot.bufferHealth, WithinAbs(occupied * std::exp(-1.0 / 1.5), 1e-6));
    metrics.advance(start + std::chrono::seconds(20), false);
    REQUIRE(metrics.snapshot.bufferHealth < 0.0001f);
    REQUIRE(metrics.snapshot.sweepSpeed < 0.0001f);
}

TEST_CASE("SDR Sweep output-slot loss counts only overwritten and expired completed frames",
          "[modules][sdr_sweep][metrics][receive]") {
    using Clock = Modules::SdrSweepImpl::Clock;
    const auto start = Clock::time_point{};
    SweepAccess sweep;
    sweep.initializeSlot(start);
    REQUIRE(sweep.completeAt(start + std::chrono::milliseconds(250)));
    REQUIRE(sweep.completeAt(start + std::chrono::milliseconds(500)));
    auto metrics = sweep.getMetrics(start + std::chrono::milliseconds(500));
    REQUIRE(metrics.bufferLoss == 0.5);
    REQUIRE(metrics.bufferHealth > 0.0f);
    REQUIRE(metrics.bufferHealth < 1.0f);
    REQUIRE(sweep.consumeAt(start + std::chrono::milliseconds(750)) == Result::SUCCESS);
    REQUIRE(sweep.completeAt(start + std::chrono::seconds(1)));
    metrics = sweep.getMetrics(start + std::chrono::milliseconds(1250));
    REQUIRE_THAT(metrics.bufferLoss, WithinAbs(1.0 / 3.0, 1e-12));

    // Expire the third frame at its deadline, not at the later metrics poll.
    metrics = sweep.getMetrics(start + std::chrono::seconds(4));
    REQUIRE_THAT(metrics.bufferLoss, WithinAbs(2.0 / 3.0, 1e-12));
    REQUIRE(sweep.getProgress().expiredFrames == 1);
    REQUIRE(sweep.consumeAt(start + std::chrono::seconds(4)) == Result::YIELD);
    metrics = sweep.getMetrics(start + std::chrono::seconds(20));
    REQUIRE_THAT(metrics.bufferLoss, WithinAbs(2.0 / 3.0, 1e-12));
    REQUIRE(metrics.bufferHealth < 0.0001f);
    REQUIRE(sweep.getProgress().expiredFrames == 1);
    REQUIRE(sweep.destroy() == Result::SUCCESS);
    const auto stopped = sweep.getMetrics();
    REQUIRE(stopped.bufferHealth == 0.0f);
    REQUIRE(stopped.bufferLoss == 0.0);
    REQUIRE(stopped.sweepSpeed == 0.0f);
    REQUIRE(stopped.peakSweepSpeed == 0.0f);
    REQUIRE_FALSE(sweep.completeAt(Clock::now()));
}

TEST_CASE("SDR Sweep completed-frame speed includes the first hop and single-hop synthesis",
          "[modules][sdr_sweep][metrics][rate]") {
    using Clock = Modules::SdrSweepImpl::Clock;
    const auto start = Clock::time_point{};
    SweepAccess sweep;
    sweep.initializeSlot(start);
    // One complete frame after two seconds of acquisition and synthesis, even
    // when the legacy single-hop duration is zero.
    REQUIRE(sweep.getMetrics(start + std::chrono::seconds(1)).sweepSpeed == 0.0f);
    REQUIRE(sweep.completeAt(start + std::chrono::seconds(2)));
    auto metrics = sweep.getMetrics(start + std::chrono::seconds(2));
    REQUIRE_THAT(metrics.sweepSpeed, WithinAbs(0.5 * (1.0 - std::exp(-2.0 / 1.5)), 1e-6));
    REQUIRE(metrics.sweepSpeed > 0.0f);
    REQUIRE(metrics.peakSweepSpeed == metrics.sweepSpeed);
    REQUIRE(std::isfinite(metrics.sweepSpeed));
    const F32 first = metrics.sweepSpeed;
    metrics = sweep.getMetrics(start + std::chrono::seconds(8));
    REQUIRE(metrics.sweepSpeed < first);
    REQUIRE(metrics.peakSweepSpeed == first);
    sweep.fail();
    REQUIRE(sweep.getMetrics(start + std::chrono::milliseconds(8001)).sweepSpeed == 0.0f);

    SweepAccess slow;
    slow.initializeSlot(start);
    for (U64 i = 1; i < 32; ++i) {
        REQUIRE(slow.getMetrics(start + std::chrono::milliseconds(250 * i)).sweepSpeed == 0.0f);
    }
    REQUIRE(slow.completeAt(start + std::chrono::seconds(8)));
    const auto measured = slow.getMetrics(start + std::chrono::seconds(8));
    REQUIRE_THAT(measured.sweepSpeed, WithinAbs(0.125 * (1.0 - std::exp(-8.0 / 1.5)), 1e-6));

    SweepAccess quick;
    quick.initializeSlot(start);
    REQUIRE(quick.completeAt(start + std::chrono::milliseconds(20)));
    const auto quickMetrics = quick.getMetrics(start + std::chrono::milliseconds(250));
    REQUIRE(quickMetrics.sweepSpeed > 0.0f);
    REQUIRE(std::isfinite(quickMetrics.sweepSpeed));
}

TEST_CASE("SDR Sweep metrics throttle fast completions and reset without zero-duration division",
          "[modules][sdr_sweep][metrics][rate]") {
    using Clock = Modules::SdrSweepImpl::Clock;
    const auto start = Clock::time_point{};
    SweepAccess::MetricState metrics;
    metrics.reset(start);
    F32 previous = 0.0f;
    for (U64 i = 1; i <= 40; ++i) {
        metrics.publish(start + std::chrono::milliseconds(25 * i), i > 1);
        if (i % 10 != 0) {
            REQUIRE(metrics.snapshot.sweepSpeed == previous);
        } else {
            REQUIRE(metrics.snapshot.sweepSpeed > previous);
            previous = metrics.snapshot.sweepSpeed;
        }
    }
    REQUIRE_THAT(metrics.snapshot.sweepSpeed, WithinAbs(40.0 * (1.0 - std::exp(-1.0 / 1.5)), 1e-5));
    REQUIRE(metrics.producedFrames == 40);
    REQUIRE(metrics.droppedFrames == 39);
    REQUIRE_THAT(metrics.snapshot.bufferLoss, WithinAbs(39.0 / 40.0, 1e-12));
    metrics.advance(start + std::chrono::milliseconds(1001), false, false);
    REQUIRE(metrics.snapshot.sweepSpeed == 0.0f);

    metrics.reset(start + std::chrono::seconds(2));
    REQUIRE(metrics.producedFrames == 0);
    REQUIRE(metrics.droppedFrames == 0);
    REQUIRE(metrics.snapshot.bufferHealth == 0.0f);
    REQUIRE(metrics.snapshot.bufferLoss == 0.0);
    REQUIRE(metrics.snapshot.peakSweepSpeed == 0.0f);
    metrics.publish(start + std::chrono::seconds(2), false);
    metrics.advance(start + std::chrono::seconds(2), true);
    REQUIRE(metrics.snapshot.sweepSpeed == 0.0f);
    REQUIRE(std::isfinite(metrics.snapshot.sweepSpeed));
    metrics.advance(start + std::chrono::milliseconds(2250), true);
    REQUIRE(std::isfinite(metrics.snapshot.sweepSpeed));
    REQUIRE(metrics.snapshot.sweepSpeed == 0.0f);
    metrics.publish(start + std::chrono::milliseconds(2250), true);
    metrics.advance(start + std::chrono::milliseconds(2500), true);
    REQUIRE(metrics.snapshot.sweepSpeed > 0.0f);
}

TEST_CASE("SDR Sweep steady completion rates are independent of presentation polling",
          "[modules][sdr_sweep][metrics][rate]") {
    using Clock = Modules::SdrSweepImpl::Clock;
    const auto start = Clock::time_point{};
    U64 ticksPerSweep = 4;
    F64 expected = 1.0;
    SECTION("one completed sweep per second") {}
    SECTION("one completed sweep every five seconds") {
        ticksPerSweep = 20;
        expected = 0.2;
    }
    SweepAccess::MetricState frequent;
    SweepAccess::MetricState sparse;
    frequent.reset(start);
    sparse.reset(start);
    const U64 lastTick = 20 * ticksPerSweep;
    for (U64 tick = 1; tick <= lastTick; ++tick) {
        const auto now = start + std::chrono::milliseconds(250 * tick);
        if (tick % ticksPerSweep == 0) {
            frequent.publish(now, false);
            sparse.publish(now, false);
            REQUIRE(frequent.snapshot.sweepSpeed == sparse.snapshot.sweepSpeed);
        }
        // The same producer timeline with extra UI/consume polls in only one
        // case. Each frame is consumed immediately, so the slot stays empty.
        frequent.advance(now, false);
    }
    REQUIRE(frequent.producedFrames == 20);
    REQUIRE(sparse.producedFrames == 20);
    REQUIRE_THAT(frequent.snapshot.sweepSpeed, WithinAbs(expected, 0.001));
    REQUIRE_THAT(sparse.snapshot.sweepSpeed, WithinAbs(expected, 0.001));

    // Ordinary time before the next expected completion must not decay the
    // estimate. Compare both presentations at the same between-frame point.
    for (U64 tick = 1; tick < ticksPerSweep; ++tick) {
        frequent.advance(start + std::chrono::milliseconds(250 * (lastTick + tick)), false);
    }
    const auto between = start + std::chrono::milliseconds(250 * (lastTick + ticksPerSweep - 1));
    sparse.advance(between, false);
    REQUIRE(frequent.snapshot.sweepSpeed == sparse.snapshot.sweepSpeed);
    REQUIRE_THAT(frequent.snapshot.sweepSpeed, WithinAbs(expected, 0.001));

    const F64 estimate = frequent.speed;
    const auto idle = between + std::chrono::seconds(60);
    for (U64 tick = 1; tick <= 240; ++tick) {
        frequent.advance(between + std::chrono::milliseconds(250 * tick), false);
    }
    sparse.advance(idle, false);
    REQUIRE(frequent.snapshot.sweepSpeed == sparse.snapshot.sweepSpeed);
    REQUIRE(frequent.snapshot.sweepSpeed < 0.0001f);
    REQUIRE(frequent.speed == estimate);
    REQUIRE_THAT(frequent.snapshot.peakSweepSpeed, WithinAbs(expected, 0.001));
    frequent.advance(idle + std::chrono::milliseconds(1), false, false);
    REQUIRE(frequent.snapshot.sweepSpeed == 0.0f);
    frequent.reset(idle + std::chrono::seconds(1));
    REQUIRE(frequent.snapshot.sweepSpeed == 0.0f);
    REQUIRE(frequent.snapshot.peakSweepSpeed == 0.0f);
    REQUIRE(frequent.producedFrames == 0);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep module rejects invalid sweeps before opening hardware",
                 "[modules][sdr_sweep][validation]") {
    const auto reject = [](const Modules::SdrSweep& config) {
        const auto module = BuildSweepModule();
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE(module->state() == Module::State::ERRORED);
    };
    auto config = ScriptedSweep();
    config.endFrequency = config.startFrequency;
    reject(config);
    config = ScriptedSweep();
    config.resolution = -1.0;
    reject(config);
    config = ScriptedSweep();
    config.dwell = 0;
    reject(config);
    config = ScriptedSweep();
    config.settlingTime = -1.0f;
    reject(config);
    REQUIRE(TestScript().opens == 0);

    config = ScriptedSweep();
    config.deviceString.clear();
    const auto module = BuildSweepModule();
    REQUIRE(module->create("test", config, {}) == Result::INCOMPLETE);
    REQUIRE(TestScript().opens == 0);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep rejects unrepresentable settling guards before acquisition",
                 "[modules][sdr_sweep][validation][limits]") {
    auto config = ScriptedSweep();
    SECTION("explicit capture rate") {
        config.sampleRate = 2.0e6;
    }
    SECTION("automatic capture rate") {
        config.sampleRate = 0.0;
    }
    config.settlingTime = std::numeric_limits<F32>::max();
    const auto module = BuildSweepModule();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->outputs().empty());
    const auto calls = TestScript().calls();
    REQUIRE(calls.size() == 3);
    REQUIRE(calls.front() == "open");
    REQUIRE(calls[1].starts_with("start:"));
    REQUIRE(calls.back() == "close");
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep accepts ordinary settling guards",
                 "[modules][sdr_sweep][validation][limits]") {
    auto config = ScriptedSweep();
    config.settlingTime = 5.0f;
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    REQUIRE(Radio::GuardSamples(config.settlingTime / 1e3, 2.0e6) == 10000);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep forwards settling overrides into method selection",
                 "[modules][sdr_sweep][validation][hackrf][settling]") {
    TestScript().firmware = HackRfCaps().firmwareSweep;
    auto config = ScriptedSweep();
    config.settlingTime = 10.0f;
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    REQUIRE(sweep->sweepPlan().mode == Mode::Retune);
    REQUIRE(sweep->settlingTime == config.settlingTime);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep module ignores legacy method selections and preserves an automatic target",
                 "[modules][sdr_sweep][config][legacy][hackrf]") {
    TestScript().budget = 0;
    TestScript().firmware = HackRfCaps().firmwareSweep;
    auto config = ScriptedSweep();
    config.startFrequency = 88.0e6;
    config.endFrequency = 108.0e6;
    config.sampleRate = 20.0e6;
    config.resolution = 1.0e3;
    Parser::Map legacy;
    REQUIRE(config.serialize(legacy) == Result::SUCCESS);
    REQUIRE_FALSE(legacy.contains("sweepMethod"));
    SECTION("old explicit firmware selection") {
        legacy["sweepMethod"] = "hackrf_firmware";
    }
    SECTION("old unrecognized method") {
        legacy["sweepMethod"] = "teleport";
    }
    REQUIRE(config.deserialize(legacy) == Result::SUCCESS);
    Parser::Map saved;
    REQUIRE(config.serialize(saved) == Result::SUCCESS);
    REQUIRE_FALSE(saved.contains("sweepMethod"));
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    REQUIRE(sweep->sweepPlan().mode == Mode::Retune);
    REQUIRE(sweep->sweepPlan().segmentLength == 20000);
    REQUIRE(sweep->sweepPlan().binWidth == config.resolution);
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(std::any_cast<std::string>(output.attribute("sweepMethod")) == "retune");
}

TEST_CASE_METHOD(ScriptedFlowgraph,
                 "SDR Sweep loads old graph method entries without exposing or honoring a selector",
                 "[modules][sdr_sweep][block][config][legacy]") {
    TestScript().budget = 0;
    Blocks::SdrSweep config;
    config.deviceString = Selector;
    config.startFrequency = 100.0e6;
    config.endFrequency = 104.0e6;
    Parser::Map legacy;
    REQUIRE(config.serialize(legacy) == Result::SUCCESS);
    legacy["sweepMethod"] = "bladerf_quick_tune";
    REQUIRE(flowgraph->blockCreate("sweep", "sdr_sweep", legacy, {}) == Result::SUCCESS);
    const auto block = viewBlock("sweep");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE_FALSE(block.config.contains("sweepMethod"));
    REQUIRE(std::none_of(block.interfaceConfigs.begin(), block.interfaceConfigs.end(),
                         [](const auto& field) { return field.name == "sweepMethod" || field.label == "Method"; }));
    REQUIRE(std::any_cast<std::string>(block.outputs.at("signal").tensor.attribute("sweepMethod")) == "retune");
    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("sweep", metrics) == Result::SUCCESS);
    std::erase_if(metrics, [](const auto& metric) {
        return Parser::Get<std::string>(metric.format, "visibility") == "internal";
    });
    REQUIRE(metrics.size() == 3);
    REQUIRE(metrics[0].label == "Buffer Health");
    REQUIRE(metrics[1].label == "Buffer Loss");
    REQUIRE(metrics[2].label == "Sweep Speed");
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep module releases the radio when a plan cannot run",
                 "[modules][sdr_sweep][lifecycle]") {
    auto config = ScriptedSweep();
    SECTION("hops outside the tunable range") {
        config.startFrequency = 5.0e6;
        config.endFrequency = 9.0e6;
    }
    SECTION("startup failure") {
        TestScript().failAt = "start";
    }
    const auto module = BuildSweepModule();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(TestScript().calls().back() == "close");
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep module emits complete synthetic sweeps with grid metadata",
                 "[modules][sdr_sweep][metadata][receive]") {
    const auto config = ScriptedSweep();
    Modules::Sweep::Plan plan;
    REQUIRE(Modules::Sweep::Plan::Create({.startFrequency = config.startFrequency,
                                       .endFrequency = config.endFrequency,
                                       .resolution = config.resolution},
                                      ScriptedDevice({}).capabilities(), plan) == Result::SUCCESS);
    TestScript().reset();
    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().tones = {{.frequency = BinCenter(plan, 123), .amplitude = 0.3}};
    }

    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{1, plan.outputBins});
    SignalAxes axes;
    REQUIRE(ResolveSignalAxes(output, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{1});
    REQUIRE(axes.batch == Index{0});
    REQUIRE(std::any_cast<F32>(output.attribute("frequency")) ==
            static_cast<F32>(plan.centerFrequency()));
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == static_cast<F32>(plan.span()));
    REQUIRE(std::any_cast<bool>(output.attribute("sweepSynthetic")));
    REQUIRE(std::any_cast<F64>(output.attribute("sweepStartFrequency")) == plan.startFrequency);
    REQUIRE(std::any_cast<F64>(output.attribute("sweepEndFrequency")) == plan.endFrequency());
    REQUIRE(std::any_cast<F64>(output.attribute("sweepBinWidth")) == plan.binWidth);
    REQUIRE(std::any_cast<F64>(output.attribute("sweepCaptureRate")) == 2.0e6);
    REQUIRE(std::any_cast<std::string>(output.attribute("sweepMethod")) == "retune");

    auto* scheduler = module->context()->scheduler().get();
    REQUIRE(WaitFor([&] { return scheduler->hasPendingCompute() == Result::SUCCESS; },
                    std::chrono::seconds(10)));
    const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    const auto firstSweep = std::any_cast<U64>(output.attribute("sweepId"));
    REQUIRE(firstSweep >= 1);

    std::vector<CF32> frame(output.data<CF32>(), output.data<CF32>() + output.size());
    const auto decibels = SpectrumAnalyzerDb(frame);
    const auto reference = SpectrumAnalyzerDb(DirectTone(plan.outputBins, 123, 0.3));
    REQUIRE(ArgMax(decibels) == 123);
    REQUIRE_THAT(decibels[123], WithinAbs(reference[123], 0.05));

    // One complete sweep per output, and a consumed sweep is never repeated.
    REQUIRE(WaitFor([&] { return scheduler->hasPendingCompute() == Result::SUCCESS; },
                    std::chrono::seconds(10)));
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(output.attribute("sweepId")) > firstSweep);
    const auto progress = module->getImpl<Modules::SdrSweepImpl>()->getProgress();
    REQUIRE(progress.totalHops == plan.hops.size());
    REQUIRE(progress.sweeps >= 1);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep module drops sweeps with failed hops and stops on persistent failures",
                 "[modules][sdr_sweep][receive]") {
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", ScriptedSweep(), {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    REQUIRE(WaitFor([&] { return sweep->getProgress().sweeps >= 1; }, std::chrono::seconds(10)));

    // The first hop fails to tune repeatedly, so no sweep can complete.
    const auto firstHop = sweep->sweepPlan().hops.front().frequency;
    TestScript().failAt = jst::fmt::format("tune:{:.0f}", firstHop / 1e6);
    REQUIRE(WaitFor([&] {
        return module->context()->scheduler()->hasPendingCompute() == Result::ERROR;
    }, std::chrono::seconds(10)));
    REQUIRE(sweep->getProgress().failedHops >= 3);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep keeps one receive session across hops and sweeps",
                 "[modules][sdr_sweep][receive][retune]") {
    const auto module = BuildSweepModule();
    REQUIRE(module->create("test", ScriptedSweep(), {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    const U64 hops = sweep->sweepPlan().hops.size();
    REQUIRE(hops >= 3);
    REQUIRE(WaitFor([&] { return sweep->getProgress().sweeps >= 3; }, std::chrono::seconds(10)));
    REQUIRE(module->destroy() == Result::SUCCESS);

    const auto calls = TestScript().calls();
    const auto count = [&](const std::string& prefix) {
        return std::count_if(calls.begin(), calls.end(), [&](const auto& call) {
            return call.starts_with(prefix);
        });
    };
    REQUIRE(count("start:") == 1);
    REQUIRE(count("stop") == 1);
    REQUIRE(count("tune:") >= static_cast<I64>(3 * hops));
    REQUIRE(calls.end()[-2] == "stop");
    REQUIRE(calls.back() == "close");
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep retries a failing later hop in place and stops after bounded failures",
                 "[modules][sdr_sweep][receive][failure]") {
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", ScriptedSweep(), {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    REQUIRE(WaitFor([&] { return sweep->getProgress().sweeps >= 1; }, std::chrono::seconds(10)));

    // Earlier hops still succeed, so their success must not reset the count.
    const auto lastHop = jst::fmt::format("tune:{:.0f}",
                                          sweep->sweepPlan().hops.back().frequency / 1e6);
    TestScript().failAt = lastHop;
    REQUIRE(WaitFor([&] {
        return module->context()->scheduler()->hasPendingCompute() == Result::ERROR;
    }, std::chrono::seconds(10)));
    REQUIRE(sweep->getProgress().failedHops == 3);

    auto tunes = TestScript().calls();
    std::erase_if(tunes, [](const auto& call) { return !call.starts_with("tune:"); });
    REQUIRE(tunes.size() >= 3);
    REQUIRE(std::all_of(tunes.end() - 3, tunes.end(), [&](const auto& call) {
        return call == lastHop;
    }));
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep stops at once when the radio disconnects",
                 "[modules][sdr_sweep][receive][lifecycle]") {
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", ScriptedSweep(), {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    REQUIRE(WaitFor([&] { return sweep->getProgress().sweeps >= 1; }, std::chrono::seconds(10)));
    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().nextRead = Radio::Device::Capture{.status = Radio::Device::Status::Disconnected,
                                                       .error = "unplugged"};
    }
    REQUIRE(WaitFor([&] {
        return module->context()->scheduler()->hasPendingCompute() == Result::ERROR;
    }, std::chrono::seconds(10)));
    REQUIRE(sweep->getProgress().failedHops == 0);
    REQUIRE(module->destroy() == Result::SUCCESS);
    REQUIRE(TestScript().calls().back() == "close");
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep frames expire after stalls",
                 "[modules][sdr_sweep][receive]") {
    TestScript().budget = 0;
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", ScriptedSweep(), {}) == Result::SUCCESS);
    auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    using Clock = Modules::SdrSweepImpl::Clock;
    const auto now = Clock::now();
    const SweepAccess::Frame frame{.sweep = 1, .started = now, .completed = now};
    REQUIRE_FALSE(SweepAccess::Expired(*sweep, frame, now + std::chrono::milliseconds(1999)));
    REQUIRE(SweepAccess::Expired(*sweep, frame, now + std::chrono::milliseconds(2001)));

    // Slow sweeps extend the lifetime to twice their duration.
    (sweep->*SweepAccess::sweepDurationMember()).publish(5.0);
    REQUIRE_FALSE(SweepAccess::Expired(*sweep, frame, now + std::chrono::seconds(9)));
    REQUIRE(SweepAccess::Expired(*sweep, frame, now + std::chrono::seconds(11)));
}

TEST_CASE_METHOD(ScriptedFlowgraph,
                 "SDR Sweep connects directly to the Spectrum Analyzer",
                 "[modules][sdr_sweep][block][integration]") {
    Blocks::SdrSweep sweep;
    sweep.deviceString = Selector;
    sweep.startFrequency = 100.0e6;
    sweep.endFrequency = 104.0e6;
    sweep.resolution = 10.0e3;
    REQUIRE(flowgraph->blockCreate("sweep", sweep, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("sweep").state == Block::State::Created);

    TensorMap inputs;
    inputs["buffer"].requested("sweep", "signal");
    Blocks::SpectrumAnalyzer analyzer;
    analyzer.waterfallHeight = 64;
    REQUIRE(flowgraph->blockCreate("analyzer", analyzer, inputs) == Result::SUCCESS);
    const auto view = viewBlock("analyzer");
    INFO(view.diagnostic);
    REQUIRE(view.state == Block::State::Created);
}

//
// Numeric limits.
//

TEST_CASE("SDR Sweep rejects extreme finite inputs before any conversion",
          "[modules][sdr_sweep][plan][limits]") {
    Modules::Sweep::Plan plan;
    const auto reject = [&](const Modules::Sweep::Plan::Request& request, const Radio::Device::Capabilities& caps) {
        REQUIRE(Modules::Sweep::Plan::Create(request, caps, plan) == Result::ERROR);
        REQUIRE(plan.hops.empty());
        REQUIRE(plan.outputBins == 0);
    };

    // A tiny physical span cannot contain the minimum output grid.
    reject({.startFrequency = 0.0, .endFrequency = 1e-300, .resolution = 1e-300}, TestCaps());
    const auto fine = RequirePlan({.startFrequency = 100.0e6, .endFrequency = 101.0e6,
                                   .resolution = std::numeric_limits<F64>::denorm_min()}, TestCaps());
    REQUIRE(fine.segmentLength == Modules::Sweep::MaxSegmentLength);
    // A huge span with a huge resolution needs too few capture points.
    reject({.startFrequency = 1e300, .endFrequency = 1.5e300, .resolution = 1e299}, TestCaps());
    // A huge span at a normal resolution needs too many bins.
    reject({.startFrequency = 100.0e6, .endFrequency = 1e300, .resolution = 10.0e3}, TestCaps());

    // Firmware planning bounds its MHz units before converting them.
    const auto caps = HackRfCaps();
    const auto firmware = Mode::Firmware;
    reject({.startFrequency = 1e300, .endFrequency = 2e300, .resolution = 10.0e3,
            .mode = firmware}, caps);
    reject({.startFrequency = 88.0e6, .endFrequency = 1e300, .resolution = 10.0e3,
            .mode = firmware}, caps);
    const auto firmwareFine = RequirePlan({.startFrequency = 88.0e6, .endFrequency = 108.0e6,
                                           .resolution = 1e-300, .mode = firmware}, caps);
    REQUIRE(firmwareFine.segmentLength == caps.firmwareSweep->maxCaptureLength);
    REQUIRE(Modules::Sweep::Plan::ResolveMode({.resolution = 1e-300}, caps) == firmware);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep module rejects unrepresentable resolutions without allocating",
                 "[modules][sdr_sweep][validation][limits]") {
    auto config = ScriptedSweep();
    config.startFrequency = 0.0;
    config.endFrequency = 1e-300;
    config.resolution = 1e-300;
    const auto module = BuildSweepModule();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->outputs().empty());
    REQUIRE(TestScript().calls() == std::vector<std::string>{"open", "close"});
}

//
// Applied capture rate.
//

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep plans its grid from the capture rate the radio applied",
                 "[modules][sdr_sweep][metadata][rate]") {
    // The radio applies 2.1 MS/s when asked for 2 MS/s.
    TestScript().rateOffset = 0.1e6;
    const auto config = ScriptedSweep();
    auto caps = ScriptedDevice({}).capabilities();
    const auto plan = RequirePlan({.startFrequency = config.startFrequency,
                                   .endFrequency = config.endFrequency,
                                   .sampleRate = 2.1e6,
                                   .resolution = config.resolution}, caps);
    const F64 rate = TestScript().rateOffset;
    TestScript().reset();
    TestScript().rateOffset = rate;
    {
        std::lock_guard lock(TestScript().mutex);
        TestScript().tones = {{.frequency = BinCenter(plan, 77), .amplitude = 0.3}};
    }

    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    REQUIRE(sweep->sweepPlan().sampleRate == 2.1e6);
    REQUIRE(sweep->sweepPlan().segmentLength == plan.segmentLength);
    REQUIRE(sweep->sweepPlan().outputBins == plan.outputBins);
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(output.shape() == Shape{1, plan.outputBins});
    REQUIRE(std::any_cast<F64>(output.attribute("sweepCaptureRate")) == 2.1e6);
    REQUIRE(std::any_cast<F64>(output.attribute("sweepBinWidth")) == plan.binWidth);
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == static_cast<F32>(plan.span()));

    auto* scheduler = module->context()->scheduler().get();
    REQUIRE(WaitFor([&] { return scheduler->hasPendingCompute() == Result::SUCCESS; },
                    std::chrono::seconds(10)));
    const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    const std::vector<CF32> frame(output.data<CF32>(), output.data<CF32>() + output.size());
    const auto decibels = SpectrumAnalyzerDb(frame);
    const auto reference = SpectrumAnalyzerDb(DirectTone(plan.outputBins, 77, 0.3));
    REQUIRE(ArgMax(decibels) == 77);
    REQUIRE_THAT(decibels[77], WithinAbs(reference[77], 0.05));
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep replans a bounded effective grid after rate readback",
                 "[modules][sdr_sweep][metadata][adaptive][rate]") {
    TestScript().budget = 0;
    TestScript().rateOffset = 0.123e6;
    auto config = ScriptedSweep();
    config.endFrequency = config.startFrequency + 1456.0e6;
    config.resolution = 5.0e3;
    const auto module = BuildSweepModule();
    const SweepCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    const auto& plan = sweep->sweepPlan();
    const auto expected = RequirePlan({.startFrequency = config.startFrequency,
                                       .endFrequency = config.endFrequency,
                                       .sampleRate = 2.123e6, .resolution = config.resolution}, TestCaps());
    REQUIRE(plan.segmentLength == expected.segmentLength);
    REQUIRE(plan.outputBins == expected.outputBins);
    REQUIRE(plan.outputBins <= Modules::Sweep::MaxOutputBins);
    REQUIRE(plan.endFrequency() >= config.endFrequency);
    REQUIRE(plan.binWidth > config.resolution);
    REQUIRE(sweep->resolution == config.resolution);
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(std::any_cast<F64>(output.attribute("sweepRequestedResolution")) == config.resolution);
    REQUIRE(std::any_cast<F64>(output.attribute("sweepBinWidth")) == plan.binWidth);
    REQUIRE(std::any_cast<U64>(output.attribute("sweepOutputBins")) == plan.outputBins);
    REQUIRE(output.shape() == Shape{1, plan.outputBins});
}

TEST_CASE_METHOD(ScriptedFlowgraph,
                 "SDR Sweep retains effective metadata and recovers the saved target",
                 "[modules][sdr_sweep][block][adaptive]") {
    TestScript().budget = 0;
    Blocks::SdrSweep config;
    config.deviceString = Selector;
    config.startFrequency = 100.0e6;
    config.endFrequency = 1556.0e6;
    config.resolution = 5.0e3;
    REQUIRE(flowgraph->blockCreate("sweep", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("sweep").state == Block::State::Created);
    const auto wide = viewBlock("sweep").outputs.at("signal").tensor;
    REQUIRE_THAT(std::any_cast<F64>(wide.attribute("sweepBinWidth")), WithinAbs(5555.555556, 1e-6));
    REQUIRE(std::any_cast<U64>(wide.attribute("sweepOutputBins")) == 262080);
    REQUIRE(std::any_cast<F64>(wide.attribute("sweepRequestedResolution")) == config.resolution);
    std::vector<Flowgraph::View::MetricEntry> metrics;
    REQUIRE(flowgraph->view().metrics("sweep", metrics) == Result::SUCCESS);
    std::erase_if(metrics, [](const auto& metric) {
        return Parser::Get<std::string>(metric.format, "visibility") == "internal";
    });
    REQUIRE(metrics.size() == 3);
    REQUIRE(metrics[0].label == "Buffer Health");
    REQUIRE(metrics[1].label == "Buffer Loss");
    REQUIRE(metrics[2].label == "Sweep Speed");
    REQUIRE(std::any_cast<std::pair<std::string, F32>>(metrics[2].value).first == "0.0 sweeps/s");
    auto saved = viewBlock("sweep").config;
    REQUIRE(Parser::Get<F64>(saved, "resolution") == config.resolution);
    Blocks::SdrSweep restored;
    REQUIRE(restored.deserialize(saved) == Result::SUCCESS);
    REQUIRE(restored.resolution == config.resolution);

    REQUIRE(flowgraph->blockReconfigure("sweep", {{"endFrequency", 120.0e6}}) == Result::SUCCESS);
    REQUIRE(viewBlock("sweep").state == Block::State::Created);
    const auto narrow = viewBlock("sweep").outputs.at("signal").tensor;
    REQUIRE(std::any_cast<F64>(narrow.attribute("sweepBinWidth")) == config.resolution);
    REQUIRE(std::any_cast<U64>(narrow.attribute("sweepOutputBins")) == 4000);
    REQUIRE(Parser::Get<F64>(viewBlock("sweep").config, "resolution") == config.resolution);
}

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep rejects a firmware sweep the radio cannot run at its fixed rate",
                 "[modules][sdr_sweep][rate][hackrf]") {
    TestScript().firmware = HackRfCaps().firmwareSweep;
    TestScript().rateOffset = 1.0;
    auto config = ScriptedSweep();
    config.startFrequency = 88.0e6;
    config.endFrequency = 108.0e6;
    const auto module = BuildSweepModule();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(TestScript().calls().back() == "close");
}

//
// Cancellation.
//

TEST_CASE_METHOD(ScriptedDrivers,
                 "SDR Sweep stops promptly during long captures and reads in bounded chunks",
                 "[modules][sdr_sweep][lifecycle][cancellation]") {
    auto config = ScriptedSweep();
    config.resolution = 1.0e3;
    config.dwell = Modules::Sweep::MaxDwell;

    SECTION("bounded read requests") {
        const auto module = BuildSweepModule();
        const SweepCleanup cleanup{module};
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        // One hop needs 128,000 samples, far above a single read chunk.
        REQUIRE(module->getImpl<Modules::SdrSweepImpl>()->sweepPlan().captureSamples() == 128000);
        REQUIRE(WaitFor([] { return TestScript().largestRead > 0; }));
        REQUIRE(WaitFor([&] {
            return module->getImpl<Modules::SdrSweepImpl>()->getProgress().completedHops > 0;
        }, std::chrono::seconds(10)));
        REQUIRE(TestScript().largestRead <= 16384);
    }

    SECTION("stop during a slow hop") {
        {
            std::lock_guard lock(TestScript().mutex);
            TestScript().readLimit = 64;
            TestScript().readDelay = std::chrono::milliseconds(2);
        }
        // A hop now takes 2,000 reads of at least 2 ms each.
        const auto module = BuildSweepModule();
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(WaitFor([] { return TestScript().largestRead > 0; }));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto start = std::chrono::steady_clock::now();
        REQUIRE(module->destroy() == Result::SUCCESS);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        REQUIRE(elapsed < std::chrono::milliseconds(500));
        REQUIRE(TestScript().calls().back() == "close");
        REQUIRE(module->getImpl<Modules::SdrSweepImpl>()->getProgress().sweeps == 0);
    }
}

//
// HackRF firmware stream decoding.
//

namespace {

constexpr U64 TestBlockBytes = 64;

std::vector<U8> SweepBlock(const U64 frequency, const I8 value) {
    std::vector<U8> block(TestBlockBytes, static_cast<U8>(value));
    block[0] = 0x7f;
    block[1] = 0x7f;
    for (U64 i = 0; i < 8; ++i) {
        block[2 + i] = static_cast<U8>(frequency >> (8 * i));
    }
    return block;
}

Radio::HackRfTransferQueue::Transfer SweepTransfer(const std::vector<std::vector<U8>>& blocks,
                                             const U64 partialBytes = 0,
                                             const bool gapBefore = false) {
    Radio::HackRfTransferQueue::Transfer transfer{.gapBefore = gapBefore};
    for (const auto& block : blocks) {
        transfer.bytes.insert(transfer.bytes.end(), block.begin(), block.end());
    }
    transfer.bytes.insert(transfer.bytes.end(), partialBytes, 0x7f);
    return transfer;
}

// Decodes transfers into the real assembler and returns the payload marker of
// every hop of each completed sweep.
std::vector<std::vector<I32>> DecodeSweeps(const std::vector<Radio::HackRfTransferQueue::Transfer>& transfers) {
    const std::vector<U64> headers{10, 20, 30};
    Modules::Sweep::Plan plan;
    plan.mode = Mode::Firmware;
    plan.sampleRate = 4.0;
    plan.segmentLength = 4;
    plan.binWidth = 1.0;
    plan.outputBins = 12;
    for (U64 hop = 0; hop < headers.size(); ++hop) {
        plan.hops.push_back({.frequency = static_cast<F64>(headers[hop]),
                             .runs = {{.segment = 0, .grid = hop * 4, .length = 4}}});
    }
    Modules::Sweep::Assembler assembler;
    REQUIRE(assembler.configure(plan) == Result::SUCCESS);
    std::vector<CF32> frame(plan.outputBins);

    Radio::HackRfSweepDecoder decoder;
    decoder.configure(headers, 1, 4, TestBlockBytes, 0.0);
    std::vector<std::vector<I32>> sweeps;
    for (const auto& transfer : transfers) {
        REQUIRE(decoder.decode(transfer, [&](const Radio::Device::SweepCapture& capture) {
            if (assembler.add(capture) == Result::SUCCESS && assembler.complete()) {
                std::vector<I32> markers;
                for (U64 hop = 0; hop < headers.size(); ++hop) {
                    // A constant payload lands on the center bin of each run.
                    markers.push_back(static_cast<I32>(
                        std::lround(assembler.spectrum()[hop * 4 + 2].real() * 128.0f)));
                }
                sweeps.push_back(markers);
                REQUIRE(assembler.synthesize(frame) == Result::SUCCESS);
            }
            return true;
        }));
    }
    return sweeps;
}

}  // namespace

TEST_CASE("SDR Sweep never assembles a firmware sweep across lost data",
          "[modules][sdr_sweep][hackrf][sequence]") {
    const auto a = [](const I8 value) { return SweepBlock(10, value); };
    const auto b = [](const I8 value) { return SweepBlock(20, value); };
    const auto c = [](const I8 value) { return SweepBlock(30, value); };

    SECTION("headers alone cannot reveal a whole lost cycle") {
        // Without loss tracking, hop A (1) and hops B and C (5, 6) would
        // form one sweep across the dropped transfer.
        const auto sweeps = DecodeSweeps({
            SweepTransfer({c(0), a(1)}),
            SweepTransfer({b(5), c(6), a(7), b(8), c(9)}, 0, true),
        });
        REQUIRE(sweeps == std::vector<std::vector<I32>>{{7, 8, 9}});
    }

    SECTION("a trailing partial block breaks the cycle after the blocks before it") {
        const auto sweeps = DecodeSweeps({
            SweepTransfer({c(0), a(1), b(2), c(3)}),
            SweepTransfer({a(4)}, TestBlockBytes / 2),
            SweepTransfer({b(5), c(6), a(7), b(8), c(9)}),
        });
        REQUIRE(sweeps == std::vector<std::vector<I32>>{{1, 2, 3}, {7, 8, 9}});
    }

    SECTION("malformed blocks break the cycle in place") {
        auto broken = b(5);
        broken[1] = 0;
        const auto sweeps = DecodeSweeps({
            SweepTransfer({c(0), a(1), broken, c(6), a(7), b(8), c(9)}),
        });
        REQUIRE(sweeps == std::vector<std::vector<I32>>{{7, 8, 9}});
    }
}
