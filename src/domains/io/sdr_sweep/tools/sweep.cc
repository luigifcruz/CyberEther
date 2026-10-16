#define POCKETFFT_NO_MULTITHREADING
#include "domains/dsp/fft/pocketfft.hh"

#include "sweep.hh"

#include <algorithm>
#include <cmath>
#include <new>
#include <optional>

#include "jetstream/logger.hh"

namespace Jetstream::Modules::Sweep {

namespace {

using Mode = Plan::Mode;
using Run = Plan::Run;
using Hop = Plan::Hop;
using Request = Plan::Request;
using Capabilities = Radio::Device::Capabilities;
using FirmwareSweep = Radio::Device::Firmware;
using SweepLayout = Radio::Device::Layout;
using Range = Radio::Device::Range;

constexpr F64 BinTolerance = 0.05;
constexpr U64 BlackmanGuardBins = 3;

std::optional<U64> CeilDiv(const F64 numerator, const F64 denominator, const U64 limit) {
    const F64 quotient = std::ceil(numerator / denominator - 1e-9);
    if (!std::isfinite(quotient) || quotient < 0.0 || quotient > static_cast<F64>(limit)) {
        return std::nullopt;
    }
    return static_cast<U64>(quotient);
}

void AddRun(Hop& hop, const I64 grid, const U64 segment, const U64 length, const U64 bins) {
    const I64 first = std::max<I64>(grid, 0);
    const I64 last = std::min(grid + static_cast<I64>(length), static_cast<I64>(bins));
    if (first < last) {
        hop.runs.push_back({.segment = segment + static_cast<U64>(first - grid),
                            .grid = static_cast<U64>(first),
                            .length = static_cast<U64>(last - first)});
    }
}

Result CheckCoverage(const Plan& plan) {
    std::vector<F64> coverage(plan.outputBins, 0.0);
    for (const auto& hop : plan.hops) {
        for (const auto& run : hop.runs) {
            for (U64 i = 0; i < run.length; ++i) {
                coverage[run.grid + i] += plan.weights[run.segment + i];
            }
        }
    }
    if (std::any_of(coverage.begin(), coverage.end(), [](const F64 weight) { return !(weight > 0.0); })) {
        JST_ERROR("[MODULE_SDR_SWEEP] The sweep hops do not cover every output bin.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result CheckOutputBins(const std::optional<U64> bins, const F64 needed) {
    if (!bins) {
        JST_ERROR("[MODULE_SDR_SWEEP] The sweep needs {:.3g} bins, above the limit of {}. Narrow the span "
                  "or use a coarser resolution.", needed, MaxOutputBins);
        return Result::ERROR;
    }
    if (*bins < MinOutputBins) {
        JST_ERROR("[MODULE_SDR_SWEEP] The sweep has only {} bins. Widen the span or use a finer "
                  "resolution to get at least {}.", *bins, MinOutputBins);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

struct Lengths {
    U64 base = 1;
    U64 step = 1;

    U64 count(const U64 limit) const {
        return limit >= base ? (limit - base) / step + 1 : 0;
    }

    U64 at(const U64 unit) const {
        return base + step * (unit - 1);
    }

    U64 above(const U64 length) const {
        return length <= base ? base : base + (length - base + step - 1) / step * step;
    }
};

std::optional<U64> CaptureLength(const Request& request, const F64 rate, const F64 span,
                                const U64 maximum, const Lengths& lengths) {
    const U64 limit = std::min(maximum, MaxSegmentLength);
    const U64 units = lengths.count(limit);
    const auto enough = [&](const U64 length) {
        const F64 bins = span / (rate / static_cast<F64>(length));
        return length >= MinSegmentLength && std::ceil(bins - 1e-9) >= MinOutputBins;
    };
    if (units == 0 || !enough(lengths.at(units))) {
        return std::nullopt;
    }
    U64 first = 1;
    U64 last = units;
    while (first < last) {
        const U64 middle = first + (last - first) / 2;
        if (enough(lengths.at(middle))) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    const auto requested = CeilDiv(rate, request.resolution, limit);
    U64 desired = requested ? lengths.above(*requested) : lengths.at(units);
    desired = std::max(desired, lengths.at(first));
    const auto fits = [&](const U64 length) {
        const auto bins = CeilDiv(span, rate / static_cast<F64>(length), MaxOutputBins);
        return bins && sizeof(CF32) * (length * (request.dwell + 4) + 6 * *bins) <=
                       MaxWorkingSetBytes;
    };
    if (desired >= MinSegmentLength && desired <= limit && fits(desired)) {
        return desired;
    }

    U64 low = 0;
    U64 high = units;
    while (low < high) {
        const U64 middle = low + (high - low + 1) / 2;
        if (fits(lengths.at(middle))) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low >= first ? std::optional<U64>(lengths.at(low)) : std::nullopt;
}

std::optional<U64> FirmwareCaptureLength(const Request& request, const FirmwareSweep& sweep,
                                         SweepLayout& layout) {
    if (!sweep.layout ||
        !sweep.layout(request.startFrequency, request.endFrequency, sweep.maxCaptureLength,
                      layout)) {
        return std::nullopt;
    }
    return CaptureLength(request, sweep.sampleRate, layout.endFrequency - layout.startFrequency,
                         sweep.maxCaptureLength,
                         {.base = sweep.captureBase, .step = sweep.captureStep});
}

Result PlanRetune(const Request& request, const Capabilities& caps, Plan& plan,
                  const bool quiet = false) {
    const F64 sampleRate = request.sampleRate > 0.0 ? request.sampleRate : caps.sweepSampleRate;
    if (!Range::Contains(caps.sampleRateRanges, sampleRate)) {
        if (!quiet) {
            JST_ERROR("[MODULE_SDR_SWEEP] Capture rate ({:.3f} MHz) not supported by the device.",
                      sampleRate / 1e6);
        }
        return Result::ERROR;
    }

    const auto segment = CaptureLength(request, sampleRate,
                                       request.endFrequency - request.startFrequency,
                                       MaxSegmentLength, {});
    if (!segment || *segment < MinSegmentLength) {
        if (!quiet) {
            JST_ERROR("[MODULE_SDR_SWEEP] A resolution of {:.3g} Hz at {:.3f} MS/s needs {:.3g}-point captures, "
                      "outside {} to {}.", request.resolution, sampleRate / 1e6,
                      sampleRate / request.resolution, MinSegmentLength, MaxSegmentLength);
        }
        return Result::ERROR;
    }
    const U64 length = *segment;

    const F64 binWidth = sampleRate / static_cast<F64>(length);
    const auto half = static_cast<I64>(std::floor(caps.usableBandwidth * static_cast<F64>(length) / 2.0));
    const auto dcBins = caps.dcExclusion > 0.0
        ? CeilDiv(caps.dcExclusion, binWidth, length)
        : std::optional<U64>(0);
    const I64 dc = dcBins
        ? static_cast<I64>(*dcBins + (caps.dcExclusion > 0.0 ? BlackmanGuardBins : 0))
        : half;
    if (half - dc < 1) {
        if (!quiet) {
            JST_ERROR("[MODULE_SDR_SWEEP] The usable capture bandwidth is too narrow at this resolution.");
        }
        return Result::ERROR;
    }

    const auto stitched = CeilDiv(request.endFrequency - request.startFrequency, binWidth,
                                  MaxOutputBins);
    if (!quiet) {
        JST_CHECK(CheckOutputBins(stitched, (request.endFrequency - request.startFrequency) / binWidth));
    } else if (!stitched || *stitched < MinOutputBins) {
        return Result::ERROR;
    }

    plan.sampleRate = sampleRate;
    plan.segmentLength = length;
    plan.binWidth = binWidth;
    plan.startFrequency = request.startFrequency;
    plan.outputBins = *stitched;

    const auto center = static_cast<I64>(length / 2);
    plan.weights.assign(length, 0.0f);
    for (I64 offset = -half; offset < half; ++offset) {
        if (offset >= -dc && offset < dc) {
            continue;
        }
        const F64 taper = std::cos(JST_PI * (static_cast<F64>(offset) + 0.5) / (2.0 * static_cast<F64>(half)));
        plan.weights[static_cast<U64>(center + offset)] = static_cast<F32>(taper * taper);
    }

    const U64 bins = plan.outputBins;
    const auto step = half - dc;
    std::vector<char> covered(bins, 0);
    U64 first = 0;
    for (I64 grid = half; first < bins; grid += step) {
        if (plan.hops.size() >= MaxHops) {
            if (!quiet) {
                JST_ERROR("[MODULE_SDR_SWEEP] The sweep needs more than {} hops.", MaxHops);
            }
            return Result::ERROR;
        }
        Hop hop;
        hop.frequency = std::round(plan.startFrequency + (static_cast<F64>(grid) + 0.5) * binWidth);
        if (!Range::Contains(caps.frequencyRanges, hop.frequency)) {
            if (!quiet) {
                JST_ERROR("[MODULE_SDR_SWEEP] Hop at {:.3f} MHz is outside the tunable range.",
                          hop.frequency / 1e6);
            }
            return Result::ERROR;
        }
        AddRun(hop, grid - half, static_cast<U64>(center - half), static_cast<U64>(step), bins);
        AddRun(hop, grid + dc, static_cast<U64>(center + dc), static_cast<U64>(step), bins);
        for (const auto& run : hop.runs) {
            std::fill_n(covered.begin() + static_cast<std::ptrdiff_t>(run.grid), run.length, 1);
        }
        plan.hops.push_back(std::move(hop));
        while (first < bins && covered[first]) {
            ++first;
        }
    }
    return CheckCoverage(plan);
}

Result PlanFirmware(const Request& request, const Capabilities& caps, Plan& plan) {
    const auto& sweep = *caps.firmwareSweep;
    if (request.sampleRate > 0.0 && request.sampleRate != sweep.sampleRate) {
        JST_ERROR("[MODULE_SDR_SWEEP] Firmware sweeps capture at {:.0f} MS/s. Set Capture Rate to 0 or "
                  "{:.0f} MHz.", sweep.sampleRate / 1e6, sweep.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (request.dwell > sweep.maxDwell) {
        JST_ERROR("[MODULE_SDR_SWEEP] Firmware sweeps support a dwell of at most {}.", sweep.maxDwell);
        return Result::ERROR;
    }

    SweepLayout layout;
    const auto segment = FirmwareCaptureLength(request, sweep, layout);
    if (!segment) {
        JST_ERROR("[MODULE_SDR_SWEEP] No firmware sweep covers {:.3f} to {:.3f} MHz at this resolution.",
                  request.startFrequency / 1e6, request.endFrequency / 1e6);
        return Result::ERROR;
    }
    const U64 length = *segment;
    if (!sweep.layout(request.startFrequency, request.endFrequency, length, layout)) {
        JST_ERROR("[MODULE_SDR_SWEEP] The firmware sweep layout rejected a {}-point capture.", length);
        return Result::ERROR;
    }
    const F64 span = layout.endFrequency - layout.startFrequency;
    const F64 binWidth = sweep.sampleRate / static_cast<F64>(length);
    const auto bins = CeilDiv(span, binWidth, MaxOutputBins);
    JST_CHECK(CheckOutputBins(bins, span / binWidth));

    plan.sampleRate = sweep.sampleRate;
    plan.segmentLength = length;
    plan.binWidth = binWidth;
    plan.startFrequency = layout.startFrequency;
    plan.outputBins = *bins;

    const U64 center = length / 2;
    const F64 origin = plan.startFrequency + 0.5 * binWidth;

    plan.weights.assign(length, 1.0f);
    for (const auto& source : layout.hops) {
        Hop hop;
        hop.frequency = source.frequency;
        for (const auto& band : source.bands) {
            const F64 offset = (hop.frequency - origin) / binWidth +
                               static_cast<F64>(band.offset) - static_cast<F64>(center);
            if (std::abs(offset - std::round(offset)) > 1e-6) {
                JST_ERROR("[MODULE_SDR_SWEEP] Firmware sweep bins do not align with the output grid.");
                return Result::ERROR;
            }
            AddRun(hop, static_cast<I64>(std::llround(offset)), band.offset, band.length,
                   plan.outputBins);
        }
        plan.hops.push_back(std::move(hop));
    }
    return CheckCoverage(plan);
}

}  // namespace

std::string Plan::ModeName(const Mode mode) {
    switch (mode) {
        case Mode::Automatic: return "auto";
        case Mode::Retune:    return "retune";
        case Mode::Firmware:  return "firmware";
    }
    return "unknown";
}

F64 Plan::endFrequency() const {
    return startFrequency + span();
}

F64 Plan::centerFrequency() const {
    return startFrequency + (static_cast<F64>(outputBins / 2) + 0.5) * binWidth;
}

F64 Plan::span() const {
    return static_cast<F64>(outputBins) * binWidth;
}

U64 Plan::captureSamples() const {
    return segmentLength * dwell;
}

U64 Plan::workingSetBytes() const {
    U64 runs = 0;
    for (const auto& hop : hops) {
        runs += hop.runs.size();
    }
    return sizeof(CF32) * (captureSamples() + 4 * segmentLength + 6 * outputBins) +
           sizeof(Hop) * hops.size() + sizeof(Run) * runs;
}

Radio::Sweep::Request Plan::deviceRequest() const {
    Radio::Sweep::Request request;
    request.firmware = mode == Mode::Firmware;
    request.sampleRate = sampleRate;
    request.captureLength = segmentLength;
    request.dwell = dwell;
    request.startFrequency = startFrequency;
    request.endFrequency = endFrequency();
    for (const auto& hop : hops) {
        request.frequencies.push_back(hop.frequency);
    }
    return request;
}

Plan::Mode Plan::ResolveMode(const Request& request, const Capabilities& caps) {
    if (request.mode != Mode::Automatic) {
        return request.mode;
    }
    if (caps.firmwareSweep) {
        const auto& sweep = *caps.firmwareSweep;
        const bool rate = request.sampleRate <= 0.0 || request.sampleRate == sweep.sampleRate;
        bool fits = true;
        bool meetsTarget = true;
        if (request.endFrequency > request.startFrequency) {
            SweepLayout layout;
            const auto length = FirmwareCaptureLength(request, sweep, layout);
            fits = length.has_value();
            meetsTarget = length && sweep.sampleRate / static_cast<F64>(*length) <= request.resolution;
        }
        if (rate && fits && request.settlingTime == 0.0 && request.dwell <= sweep.maxDwell) {
            if (!meetsTarget) {
                Plan retune;
                retune.dwell = request.dwell;
                if (PlanRetune(request, caps, retune, true) == Result::SUCCESS &&
                    retune.binWidth <= request.resolution) {
                    return Mode::Retune;
                }
            }
            return Mode::Firmware;
        }
    }
    return Mode::Retune;
}

Result Plan::Create(const Request& request, const Capabilities& caps, Plan& plan) {
    plan = {};

    if (!std::isfinite(request.startFrequency) || !std::isfinite(request.endFrequency) ||
        request.startFrequency < 0.0 || request.startFrequency >= request.endFrequency) {
        JST_ERROR("[MODULE_SDR_SWEEP] Start frequency must be finite and below the end frequency.");
        return Result::ERROR;
    }
    if (!std::isfinite(request.resolution) || request.resolution <= 0.0) {
        JST_ERROR("[MODULE_SDR_SWEEP] Resolution must be finite and positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(request.sampleRate) || request.sampleRate < 0.0) {
        JST_ERROR("[MODULE_SDR_SWEEP] Capture rate must be finite and not negative.");
        return Result::ERROR;
    }
    if (request.dwell == 0 || request.dwell > MaxDwell) {
        JST_ERROR("[MODULE_SDR_SWEEP] Dwell must be between 1 and {}.", MaxDwell);
        return Result::ERROR;
    }
    if (!std::isfinite(request.settlingTime) || request.settlingTime < 0.0) {
        JST_ERROR("[MODULE_SDR_SWEEP] Settling time must be finite and not negative.");
        return Result::ERROR;
    }

    const auto mode = ResolveMode(request, caps);
    plan.mode = mode;
    plan.dwell = request.dwell;
    if (mode == Mode::Firmware) {
        if (request.settlingTime > 0.0) {
            JST_ERROR("[MODULE_SDR_SWEEP] Firmware sweeps have fixed settling timing and cannot honor "
                      "a nonzero Settling Time override.");
            return Result::ERROR;
        }
        if (!caps.firmwareSweep) {
            JST_ERROR("[MODULE_SDR_SWEEP] Firmware sweeps are not available on this device.");
            return Result::ERROR;
        }
        JST_CHECK(PlanFirmware(request, caps, plan));
    } else {
        JST_CHECK(PlanRetune(request, caps, plan));
    }

    if (plan.workingSetBytes() > MaxWorkingSetBytes) {
        JST_ERROR("[MODULE_SDR_SWEEP] The sweep needs {:.1f} MiB of working memory, above the limit of "
                  "{} MiB. Reduce Dwell, the span, or the resolution.",
                  static_cast<F64>(plan.workingSetBytes()) / (1 << 20),
                  MaxWorkingSetBytes >> 20);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

struct Assembler::Impl {
    void finish();

    Plan plan;
    std::vector<F32> taps;
    std::vector<F32> scales;
    std::vector<U32> owners;
    std::vector<F32> power;
    std::vector<F32> capturePower;
    std::vector<CF32> spectrum;
    std::vector<CF32> transform;
    std::vector<CF32> reference;
    std::vector<CF32> reordered;
    U64 nextHop = 0;
    bool active = false;
};

Assembler::Assembler() : impl(std::make_unique<Impl>()) {}

Assembler::~Assembler() = default;

Result Assembler::configure(const Plan& plan) {
    if (plan.segmentLength == 0 || plan.outputBins == 0 || plan.hops.empty() || plan.dwell == 0 ||
        plan.weights.size() != plan.segmentLength || plan.hops.size() > MaxHops) {
        JST_ERROR("[MODULE_SDR_SWEEP] Cannot assemble an empty or inconsistent sweep plan.");
        return Result::ERROR;
    }
    const U64 length = plan.segmentLength;
    const U64 bins = plan.outputBins;
    try {
        impl->plan = plan;
        impl->spectrum.assign(bins, CF32{});
        impl->reordered.assign(bins, CF32{});
        impl->power.assign(bins, 0.0f);
        impl->transform.assign(length, CF32{});
        impl->reference.assign(length, CF32{});
        impl->capturePower.assign(length, 0.0f);

        impl->taps.resize(length);
        for (U64 i = 0; i < length; ++i) {
            const F64 position = length > 1 ? static_cast<F64>(i) / static_cast<F64>(length - 1) : 0.5;
            impl->taps[i] = static_cast<F32>(0.42 - 0.50 * std::cos(2.0 * JST_PI * position) +
                                             0.08 * std::cos(4.0 * JST_PI * position));
        }

        std::vector<F32> best(bins, 0.0f);
        impl->scales.assign(bins, 0.0f);
        impl->owners.assign(bins, 0);
        for (U64 h = 0; h < plan.hops.size(); ++h) {
            for (const auto& run : plan.hops[h].runs) {
                if (run.segment + run.length > length || run.grid + run.length > bins) {
                    JST_ERROR("[MODULE_SDR_SWEEP] Cannot assemble an empty or inconsistent sweep plan.");
                    return Result::ERROR;
                }
                for (U64 i = 0; i < run.length; ++i) {
                    const F32 weight = plan.weights[run.segment + i];
                    const U64 bin = run.grid + i;
                    impl->scales[bin] += weight;
                    if (weight > best[bin]) {
                        best[bin] = weight;
                        impl->owners[bin] = static_cast<U32>(h);
                    }
                }
            }
        }
        const F64 gain = 1.0 / (static_cast<F64>(length) * static_cast<F64>(length) *
                                static_cast<F64>(plan.dwell));
        for (auto& scale : impl->scales) {
            if (!(scale > 0.0f)) {
                JST_ERROR("[MODULE_SDR_SWEEP] The sweep hops do not cover every output bin.");
                return Result::ERROR;
            }
            scale = static_cast<F32>(gain / static_cast<F64>(scale));
        }
    } catch (const std::bad_alloc&) {
        JST_ERROR("[MODULE_SDR_SWEEP] Failed to allocate the sweep buffers.");
        return Result::ERROR;
    }
    impl->nextHop = 0;
    impl->active = false;
    return Result::SUCCESS;
}

void Assembler::begin() {
    std::fill(impl->power.begin(), impl->power.end(), 0.0f);
    std::fill(impl->spectrum.begin(), impl->spectrum.end(), CF32{});
    impl->nextHop = 0;
    impl->active = true;
}

Result Assembler::add(const Radio::Sweep::Capture& capture) {
    auto& state = *impl;
    const auto& plan = state.plan;

    if (capture.hop == 0) {
        begin();
    } else if (!state.active || capture.hop != state.nextHop) {
        state.active = false;
        return Result::INCOMPLETE;
    }

    const U64 length = plan.segmentLength;
    if (capture.hop >= plan.hops.size() || capture.samples.size() < plan.captureSamples()) {
        state.active = false;
        JST_ERROR("[MODULE_SDR_SWEEP] Hop capture does not match the sweep plan.");
        return Result::ERROR;
    }

    const auto& hop = plan.hops[capture.hop];
    if (std::abs(capture.frequency - hop.frequency) > BinTolerance * plan.binWidth) {
        state.active = false;
        JST_ERROR("[MODULE_SDR_SWEEP] Hop {} tuned to {:.3f} Hz instead of {:.3f} Hz, beyond the "
                  "{:.2f}-bin mapping tolerance.", capture.hop, capture.frequency,
                  hop.frequency, BinTolerance);
        return Result::ERROR;
    }

    const pocketfft::shape_t shape{static_cast<std::size_t>(length)};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(CF32))};
    const pocketfft::shape_t axes{0};
    const U64 center = length / 2;
    const U64 low = center > BlackmanGuardBins ? center - BlackmanGuardBins : 0;
    const U64 high = std::min(length, center + BlackmanGuardBins + 1);
    const bool removeDc = std::none_of(hop.runs.begin(), hop.runs.end(), [low, high](const auto& run) {
        return run.segment < high && low < run.segment + run.length;
    });
    for (U64 j = 0; j < plan.dwell; ++j) {
        const CF32* samples = capture.samples.data() + j * length;
        CF64 mean{};
        if (removeDc) {
            for (U64 i = 0; i < length; ++i) {
                mean += static_cast<CF64>(samples[i]);
            }
            mean /= static_cast<F64>(length);
        }
        const CF32 offset = static_cast<CF32>(mean);
        for (U64 i = 0; i < length; ++i) {
            state.transform[i] = (samples[i] - offset) * state.taps[i];
        }
        pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
                       state.transform.data(), state.transform.data(), 1.0f);
        if (j == 0) {
            std::copy(state.transform.begin(), state.transform.end(), state.reference.begin());
        }
        for (U64 i = 0; i < length; ++i) {
            const F32 value = std::norm(state.transform[i]);
            state.capturePower[i] = j == 0 ? value : state.capturePower[i] + value;
        }
    }

    const U64 shift = length - length / 2;
    for (const auto& run : hop.runs) {
        for (U64 i = 0; i < run.length; ++i) {
            const U64 segment = run.segment + i;
            const U64 bin = run.grid + i;
            const U64 unshifted = (segment + shift) % length;
            state.power[bin] += plan.weights[segment] * state.capturePower[unshifted];
            if (state.owners[bin] == capture.hop) {
                state.spectrum[bin] = state.reference[unshifted];
            }
        }
    }

    if (++state.nextHop == plan.hops.size()) {
        state.finish();
    }
    return Result::SUCCESS;
}

bool Assembler::complete() const {
    return impl->active && impl->nextHop == impl->plan.hops.size();
}

U64 Assembler::completedHops() const {
    return impl->active ? impl->nextHop : 0;
}

void Assembler::Impl::finish() {
    for (U64 k = 0; k < plan.outputBins; ++k) {
        const F32 magnitude = std::sqrt(power[k] * scales[k]);
        spectrum[k] = std::polar(magnitude, std::arg(spectrum[k]));
    }
}

Result Assembler::synthesize(std::span<CF32> output) {
    auto& state = *impl;
    const U64 bins = state.plan.outputBins;
    if (!complete() || output.size() < bins) {
        return Result::ERROR;
    }

    const U64 shift = bins - bins / 2;
    for (U64 k = 0; k < bins; ++k) {
        state.reordered[(k + shift) % bins] = state.spectrum[k];
    }
    const pocketfft::shape_t shape{static_cast<std::size_t>(bins)};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(CF32))};
    pocketfft::c2c(shape, stride, stride, pocketfft::shape_t{0}, pocketfft::BACKWARD,
                   state.reordered.data(), output.data(), 1.0f);
    state.active = false;
    return Result::SUCCESS;
}

std::span<const CF32> Assembler::spectrum() const {
    return impl->spectrum;
}

}  // namespace Jetstream::Modules::Sweep
