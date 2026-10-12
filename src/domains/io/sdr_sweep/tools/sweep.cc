#define POCKETFFT_NO_MULTITHREADING
#include "domains/dsp/fft/pocketfft.hh"

#include "sweep.hh"

#include <algorithm>
#include <cmath>
#include <optional>

#include "jetstream/logger.hh"

namespace Jetstream::Modules::Sweep {

namespace {

using Mode = Plan::Mode;
using Run = Plan::Run;
using Hop = Plan::Hop;
using Request = Plan::Request;
using Capabilities = Radio::Device::Capabilities;
using FirmwareSweep = Radio::Device::FirmwareSweep;
using SweepLayout = Radio::Device::SweepLayout;
using Range = Radio::Device::Range;

constexpr F64 BinTolerance = 0.05;

std::optional<U64> CeilDiv(const F64 numerator, const F64 denominator, const U64 limit) {
    const F64 quotient = std::ceil(numerator / denominator - 1e-9);
    if (!std::isfinite(quotient) || quotient < 0.0 || quotient > static_cast<F64>(limit)) {
        return std::nullopt;
    }
    return static_cast<U64>(quotient);
}

void AssignRun(Hop& hop, std::vector<char>& covered, const I64 grid, const U64 segment,
               const U64 length) {
    std::optional<Run> run;
    for (U64 i = 0; i < length; ++i) {
        const I64 bin = grid + static_cast<I64>(i);
        const bool usable = bin >= 0 && bin < static_cast<I64>(covered.size()) &&
                            !covered[static_cast<U64>(bin)];
        if (usable) {
            covered[static_cast<U64>(bin)] = 1;
            if (run && run->grid + run->length == static_cast<U64>(bin)) {
                ++run->length;
                continue;
            }
            if (run) {
                hop.runs.push_back(*run);
            }
            run = Run{.segment = segment + i, .grid = static_cast<U64>(bin), .length = 1};
        }
    }
    if (run) {
        hop.runs.push_back(*run);
    }
}

Result CheckOutputBins(const std::optional<U64> bins, const F64 needed) {
    if (!bins) {
        JST_ERROR("[SWEEP] The sweep needs {:.3g} bins, above the limit of {}. Narrow the span "
                  "or use a coarser resolution.", needed, MaxOutputBins);
        return Result::ERROR;
    }
    if (*bins < MinOutputBins) {
        JST_ERROR("[SWEEP] The sweep has only {} bins. Widen the span or use a finer "
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
        return bins && sizeof(CF32) * (length * (request.dwell + 3) + 4 * *bins) <=
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
            JST_ERROR("[SWEEP] Capture rate ({:.3f} MHz) not supported by the device.",
                      sampleRate / 1e6);
        }
        return Result::ERROR;
    }

    const auto segment = CaptureLength(request, sampleRate,
                                       request.endFrequency - request.startFrequency,
                                       MaxSegmentLength, {});
    if (!segment || *segment < MinSegmentLength) {
        if (!quiet) {
            JST_ERROR("[SWEEP] A resolution of {:.3g} Hz at {:.3f} MS/s needs {:.3g}-point captures, "
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
    const I64 dc = dcBins ? static_cast<I64>(*dcBins) : half;
    if (half - dc < 1) {
        if (!quiet) {
            JST_ERROR("[SWEEP] The usable capture bandwidth is too narrow at this resolution.");
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
    const U64 bins = plan.outputBins;
    std::vector<char> covered(bins, 0);
    U64 first = 0;
    while (first < bins) {
        if (plan.hops.size() >= MaxHops) {
            if (!quiet) {
                JST_ERROR("[SWEEP] The sweep needs more than {} hops.", MaxHops);
            }
            return Result::ERROR;
        }
        const I64 grid = static_cast<I64>(first) + half;
        Hop hop;
        hop.frequency = std::round(plan.startFrequency + (static_cast<F64>(grid) + 0.5) * binWidth);
        if (!Range::Contains(caps.frequencyRanges, hop.frequency)) {
            if (!quiet) {
                JST_ERROR("[SWEEP] Hop at {:.3f} MHz is outside the tunable range.",
                          hop.frequency / 1e6);
            }
            return Result::ERROR;
        }
        const U64 below = static_cast<U64>(half - dc);
        AssignRun(hop, covered, grid - half, static_cast<U64>(center - half), below);
        const I64 aboveStart = dc == 0 ? 0 : dc;
        AssignRun(hop, covered, grid + aboveStart, static_cast<U64>(center + aboveStart),
                  static_cast<U64>(half - aboveStart));
        plan.hops.push_back(std::move(hop));
        while (first < bins && covered[first]) {
            ++first;
        }
    }
    return Result::SUCCESS;
}

Result PlanFirmware(const Request& request, const Capabilities& caps, Plan& plan) {
    const auto& sweep = *caps.firmwareSweep;
    if (request.sampleRate > 0.0 && request.sampleRate != sweep.sampleRate) {
        JST_ERROR("[SWEEP] Firmware sweeps capture at {:.0f} MS/s. Set Capture Rate to 0 or "
                  "{:.0f} MHz.", sweep.sampleRate / 1e6, sweep.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (request.dwell > sweep.maxDwell) {
        JST_ERROR("[SWEEP] Firmware sweeps support a dwell of at most {}.", sweep.maxDwell);
        return Result::ERROR;
    }

    SweepLayout layout;
    const auto segment = FirmwareCaptureLength(request, sweep, layout);
    if (!segment) {
        JST_ERROR("[SWEEP] No firmware sweep covers {:.3f} to {:.3f} MHz at this resolution.",
                  request.startFrequency / 1e6, request.endFrequency / 1e6);
        return Result::ERROR;
    }
    const U64 length = *segment;
    if (!sweep.layout(request.startFrequency, request.endFrequency, length, layout)) {
        JST_ERROR("[SWEEP] The firmware sweep layout rejected a {}-point capture.", length);
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

    std::vector<char> covered(plan.outputBins, 0);
    for (const auto& source : layout.hops) {
        Hop hop;
        hop.frequency = source.frequency;
        for (const auto& band : source.bands) {
            const F64 offset = (hop.frequency - origin) / binWidth +
                               static_cast<F64>(band.offset) - static_cast<F64>(center);
            if (std::abs(offset - std::round(offset)) > 1e-6) {
                JST_ERROR("[SWEEP] Firmware sweep bins do not align with the output grid.");
                return Result::ERROR;
            }
            AssignRun(hop, covered, static_cast<I64>(std::llround(offset)), band.offset,
                      band.length);
        }
        plan.hops.push_back(std::move(hop));
    }
    if (std::find(covered.begin(), covered.end(), 0) != covered.end()) {
        JST_ERROR("[SWEEP] Firmware sweep bands do not cover the output grid.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
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
    return sizeof(CF32) * (captureSamples() + 3 * segmentLength + 4 * outputBins);
}

Radio::Device::SweepRequest Plan::deviceRequest() const {
    Radio::Device::SweepRequest request;
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
        JST_ERROR("[SWEEP] Start frequency must be finite and below the end frequency.");
        return Result::ERROR;
    }
    if (!std::isfinite(request.resolution) || request.resolution <= 0.0) {
        JST_ERROR("[SWEEP] Resolution must be finite and positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(request.sampleRate) || request.sampleRate < 0.0) {
        JST_ERROR("[SWEEP] Capture rate must be finite and not negative.");
        return Result::ERROR;
    }
    if (request.dwell == 0 || request.dwell > MaxDwell) {
        JST_ERROR("[SWEEP] Dwell must be between 1 and {}.", MaxDwell);
        return Result::ERROR;
    }
    if (!std::isfinite(request.settlingTime) || request.settlingTime < 0.0) {
        JST_ERROR("[SWEEP] Settling time must be finite and not negative.");
        return Result::ERROR;
    }

    const auto mode = ResolveMode(request, caps);
    plan.mode = mode;
    plan.dwell = request.dwell;
    plan.flatten = request.flatten;
    if (mode == Mode::Firmware) {
        if (request.settlingTime > 0.0) {
            JST_ERROR("[SWEEP] Firmware sweeps have fixed settling timing and cannot honor "
                      "a nonzero Settling Time override.");
            return Result::ERROR;
        }
        if (!caps.firmwareSweep) {
            JST_ERROR("[SWEEP] Firmware sweeps are not available on this device.");
            return Result::ERROR;
        }
        JST_CHECK(PlanFirmware(request, caps, plan));
    } else {
        JST_CHECK(PlanRetune(request, caps, plan));
    }

    if (plan.workingSetBytes() > MaxWorkingSetBytes) {
        JST_ERROR("[SWEEP] The sweep needs {:.1f} MiB of working memory, above the limit of "
                  "{} MiB. Reduce Dwell, the span, or the resolution.",
                  static_cast<F64>(plan.workingSetBytes()) / (1 << 20),
                  MaxWorkingSetBytes >> 20);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

struct Assembler::Impl {
    void flatten();

    Plan plan;
    std::vector<U64> groupStart;
    std::vector<U64> groupBins;
    std::vector<F32> levels;
    std::vector<F32> floors;
    std::vector<F64> sums;
    std::vector<U64> counts;
    std::vector<F32> shape;
    std::vector<bool> valid;
    bool shapeReady = false;
    std::vector<CF32> spectrum;
    std::vector<CF32> transform;
    std::vector<CF32> combined;
    std::vector<F32> magnitude;
    std::vector<CF32> reordered;
    U64 nextHop = 0;
    bool active = false;
};

Assembler::Assembler() : impl(std::make_unique<Impl>()) {}

Assembler::~Assembler() = default;

Result Assembler::configure(const Plan& plan) {
    if (plan.segmentLength == 0 || plan.outputBins == 0 || plan.hops.empty()) {
        JST_ERROR("[SWEEP] Cannot assemble an empty sweep plan.");
        return Result::ERROR;
    }
    try {
        impl->plan = plan;
        impl->spectrum.assign(plan.outputBins, CF32{});
        impl->reordered.assign(plan.outputBins, CF32{});
        impl->transform.assign(plan.segmentLength, CF32{});
        impl->combined.assign(plan.segmentLength, CF32{});
        impl->magnitude.assign(plan.segmentLength, 0.0f);

        const U64 length = plan.segmentLength;
        impl->groupStart.assign(length + 1, 0);
        for (const auto& hop : plan.hops) {
            for (const auto& run : hop.runs) {
                for (U64 i = 0; i < run.length; ++i) {
                    ++impl->groupStart[run.segment + i + 1];
                }
            }
        }
        for (U64 u = 0; u < length; ++u) {
            impl->groupStart[u + 1] += impl->groupStart[u];
        }
        impl->groupBins.assign(impl->groupStart.back(), 0);
        std::vector<U64> cursor(impl->groupStart.begin(), impl->groupStart.end() - 1);
        for (const auto& hop : plan.hops) {
            for (const auto& run : hop.runs) {
                for (U64 i = 0; i < run.length; ++i) {
                    impl->groupBins[cursor[run.segment + i]++] = run.grid + i;
                }
            }
        }
        impl->levels.clear();
        impl->floors.assign(length, 0.0f);
        impl->sums.assign(length + 1, 0.0);
        impl->counts.assign(length + 1, 0);
        impl->shape.assign(length, 0.0f);
        impl->valid.assign(length, false);
        impl->shapeReady = false;
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SWEEP] Failed to allocate the sweep buffers.");
        return Result::ERROR;
    }
    impl->nextHop = 0;
    impl->active = false;
    return Result::SUCCESS;
}

void Assembler::begin() {
    std::fill(impl->spectrum.begin(), impl->spectrum.end(), CF32{});
    impl->nextHop = 0;
    impl->active = true;
}

Result Assembler::add(const Radio::Device::SweepCapture& capture) {
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
        JST_ERROR("[SWEEP] Hop capture does not match the sweep plan.");
        return Result::ERROR;
    }

    const auto& hop = plan.hops[capture.hop];
    if (std::abs(capture.frequency - hop.frequency) > BinTolerance * plan.binWidth) {
        state.active = false;
        JST_ERROR("[SWEEP] Hop {} tuned to {:.3f} Hz instead of {:.3f} Hz, beyond the "
                  "{:.2f}-bin mapping tolerance.", capture.hop, capture.frequency,
                  hop.frequency, BinTolerance);
        return Result::ERROR;
    }

    const pocketfft::shape_t shape{static_cast<std::size_t>(length)};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(CF32))};
    const pocketfft::shape_t axes{0};
    for (U64 j = 0; j < plan.dwell; ++j) {
        pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
                       capture.samples.data() + j * length, state.transform.data(), 1.0f);
        if (j == 0) {
            std::copy(state.transform.begin(), state.transform.end(), state.combined.begin());
        }
        for (U64 i = 0; i < length; ++i) {
            const F32 value = std::abs(state.transform[i]);
            state.magnitude[i] = j == 0 ? value : state.magnitude[i] + value;
        }
    }
    if (plan.dwell > 1) {
        const F32 scale = 1.0f / static_cast<F32>(plan.dwell);
        for (U64 i = 0; i < length; ++i) {
            state.combined[i] = std::polar(state.magnitude[i] * scale, std::arg(state.combined[i]));
        }
    }

    const F32 gain = 1.0f / static_cast<F32>(length);
    const U64 shift = length - length / 2;
    for (const auto& run : hop.runs) {
        for (U64 i = 0; i < run.length; ++i) {
            const U64 unshifted = (run.segment + i + shift) % length;
            state.spectrum[run.grid + i] = state.combined[unshifted] * gain;
        }
    }

    ++state.nextHop;
    return Result::SUCCESS;
}

bool Assembler::complete() const {
    return impl->active && impl->nextHop == impl->plan.hops.size();
}

U64 Assembler::completedHops() const {
    return impl->active ? impl->nextHop : 0;
}

void Assembler::Impl::flatten() {
    constexpr U64 MinimumHops = 3;
    constexpr F32 MaximumCorrection = 20.0f;
    constexpr F32 Smoothing = 0.2f;
    constexpr F32 CarrierMargin = 10.0f;

    const U64 length = plan.segmentLength;
    for (U64 u = 0; u < length; ++u) {
        const U64 begin = groupStart[u];
        const U64 end = groupStart[u + 1];
        valid[u] = end - begin >= MinimumHops;
        if (!valid[u]) {
            continue;
        }
        levels.clear();
        for (U64 g = begin; g < end; ++g) {
            levels.push_back(10.0f * std::log10(std::norm(spectrum[groupBins[g]]) + 1e-30f));
        }
        const auto middle = levels.begin() + static_cast<std::ptrdiff_t>(levels.size() / 2);
        std::nth_element(levels.begin(), middle, levels.end());
        const F32 ceiling = *middle + CarrierMargin;
        F64 sum = 0.0;
        U64 count = 0;
        for (const auto level : levels) {
            if (level <= ceiling) {
                sum += level;
                ++count;
            }
        }
        floors[u] = static_cast<F32>(sum / static_cast<F64>(count));
    }

    for (U64 u = 0; u < length; ++u) {
        sums[u + 1] = sums[u] + (valid[u] ? floors[u] : 0.0);
        counts[u + 1] = counts[u] + (valid[u] ? 1 : 0);
    }
    const U64 radius = std::max<U64>(1, length / 128);
    levels.clear();
    for (U64 u = 0; u < length; ++u) {
        if (!valid[u]) {
            continue;
        }
        const U64 low = u > radius ? u - radius : 0;
        const U64 high = std::min(length, u + radius + 1);
        const F32 smoothed = static_cast<F32>((sums[high] - sums[low]) /
                                              static_cast<F64>(counts[high] - counts[low]));
        shape[u] = shapeReady ? shape[u] + Smoothing * (smoothed - shape[u]) : smoothed;
        levels.push_back(shape[u]);
    }
    shapeReady = true;
    if (levels.empty()) {
        return;
    }

    const auto middle = levels.begin() + static_cast<std::ptrdiff_t>(levels.size() / 2);
    std::nth_element(levels.begin(), middle, levels.end());
    const F32 reference = *middle;
    for (U64 u = 0; u < length; ++u) {
        if (!valid[u]) {
            continue;
        }
        const F32 correction = std::clamp(shape[u] - reference, -MaximumCorrection, MaximumCorrection);
        const F32 factor = std::pow(10.0f, -correction / 20.0f);
        for (U64 g = groupStart[u]; g < groupStart[u + 1]; ++g) {
            spectrum[groupBins[g]] *= factor;
        }
    }
}

Result Assembler::synthesize(std::span<CF32> output) {
    auto& state = *impl;
    const U64 bins = state.plan.outputBins;
    if (!complete() || output.size() < bins) {
        return Result::ERROR;
    }
    if (state.plan.flatten) {
        state.flatten();
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
