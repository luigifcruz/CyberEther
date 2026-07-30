#include <cmath>
#include <limits>

#include <jetstream/domains/dsp/filter/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <jetstream/tools/numeric.hh>

#include <jetstream/domains/dsp/filter_taps/module.hh>
#include <jetstream/domains/core/expand_dims/module.hh>
#include <jetstream/domains/core/pad/module.hh>
#include <jetstream/domains/core/unpad/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/core/multiply_constant/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/fold/module.hh>
#include <jetstream/domains/dsp/overlap_add/module.hh>
#include <jetstream/domains/core/duplicate/module.hh>

namespace Jetstream::Blocks {

namespace {

struct FilterCandidatePlan {
    bool resample = false;
    U64 padSize = 0;
    U64 resamplerOffset = 0;
    U64 resamplerSize = 0;
};

Result CalculateCandidatePlan(const Blocks::Filter& config,
                               U64 signalSize,
                               FilterCandidatePlan& candidatePlan) {
    const U64 filterSize = config.taps;
    candidatePlan.padSize = filterSize - 1;

    U64 convolutionSize = 0;
    if (!detail::CheckedAdd(filterSize, signalSize - 1, convolutionSize)) {
        JST_ERROR("[BLOCK_FILTER] Combined signal and filter extent exceeds "
                  "the supported range.");
        return Result::ERROR;
    }

    const F64 sr = config.sampleRate;
    const F64 bw = config.bandwidth;
    const F64 ct = config.center.empty() ? 0.0 : config.center.front();

    const F64 resamplerRatio = sr / bw;

    const F64 u64Limit = std::ldexp(1.0, std::numeric_limits<U64>::digits);
    if (!std::isfinite(resamplerRatio) || resamplerRatio <= 0.0 ||
        resamplerRatio >= u64Limit) {
        return Result::SUCCESS;
    }

    if (resamplerRatio != std::floor(resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER] Bypassing resampling because "
                 "filter bandwidth ({:.2f} MHz) is not a multiple "
                 "of the signal sample rate ({:.2f} MHz).",
                 bw / 1e6, sr / 1e6);
        return Result::SUCCESS;
    }

    const U64 resamplerRatioInteger = static_cast<U64>(resamplerRatio);
    if (candidatePlan.padSize % resamplerRatioInteger != 0) {
        JST_WARN("[BLOCK_FILTER] Bypassing resampling because "
                 "filter tap size minus one ({}) is not a multiple "
                 "of the resampler ratio ({}).",
                 candidatePlan.padSize,
                 resamplerRatioInteger);
        return Result::SUCCESS;
    }

    if (convolutionSize % resamplerRatioInteger != 0) {
        JST_WARN("[BLOCK_FILTER] Bypassing resampling because "
                 "filter tap size minus one ({}) plus signal "
                 "size ({}) is not a multiple of the resampler "
                 "ratio ({}).",
                 candidatePlan.padSize, signalSize,
                 resamplerRatioInteger);
        return Result::SUCCESS;
    }

    if (ct != 0.0) {
        const F64 frequencyPerBin =
            sr / static_cast<F64>(convolutionSize);
        const F64 centerBin = ct / frequencyPerBin;

        if (!std::isfinite(centerBin)) {
            JST_ERROR("[BLOCK_FILTER] Center frequency ({}) cannot be mapped "
                      "to a finite resampler bin.", ct);
            return Result::ERROR;
        }

        const F64 roundedCenterBin = std::round(centerBin);

        if (centerBin != std::floor(centerBin)) {
            JST_WARN("[BLOCK_FILTER] Output will be shifted by "
                     "{} MHz because filter center frequency "
                     "({:.2f} MHz) is not a multiple of the "
                     "frequency per bin ({} MHz).",
                     (centerBin - std::floor(centerBin)) *
                         frequencyPerBin / 1e6,
                     ct / 1e6,
                     frequencyPerBin / 1e6);
        }

        if (roundedCenterBin < 0.0) {
            const F64 centerBinMagnitude = -roundedCenterBin;
            if (!std::isfinite(centerBinMagnitude) ||
                centerBinMagnitude >= u64Limit) {
                JST_ERROR("[BLOCK_FILTER] Center frequency ({}) cannot be mapped "
                          "to a representable resampler bin.", ct);
                return Result::ERROR;
            }

            const U64 centerBinMagnitudeInteger =
                static_cast<U64>(centerBinMagnitude);
            const U64 remainder = centerBinMagnitudeInteger % convolutionSize;
            candidatePlan.resamplerOffset =
                remainder == 0 ? 0 : convolutionSize - remainder;
        } else {
            const long double convolutionExtent =
                static_cast<long double>(convolutionSize);
            const long double wrappedCenterBin = std::fmod(
                static_cast<long double>(roundedCenterBin), convolutionExtent);
            const long double u64LimitLongDouble =
                std::ldexp(1.0L, std::numeric_limits<U64>::digits);
            if (!std::isfinite(wrappedCenterBin) || wrappedCenterBin < 0.0L ||
                wrappedCenterBin >= u64LimitLongDouble) {
                JST_ERROR("[BLOCK_FILTER] Center frequency ({}) cannot be mapped "
                          "to a representable resampler bin.", ct);
                return Result::ERROR;
            }

            candidatePlan.resamplerOffset = static_cast<U64>(wrappedCenterBin);
        }
    }

    candidatePlan.resamplerSize = convolutionSize / resamplerRatioInteger;
    candidatePlan.padSize /= resamplerRatioInteger;
    candidatePlan.resample = true;

    return Result::SUCCESS;
}

}  // namespace

struct FilterImpl : public Block::Impl,
                    public DynamicConfig<Blocks::Filter> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::FilterTaps> filterTapsConfig =
        std::make_shared<Modules::FilterTaps>();
    std::shared_ptr<Modules::ExpandDims> expandDimsConfig =
        std::make_shared<Modules::ExpandDims>();
    std::shared_ptr<Modules::Pad> padSignalConfig =
        std::make_shared<Modules::Pad>();
    std::shared_ptr<Modules::Pad> padFilterConfig =
        std::make_shared<Modules::Pad>();
    std::shared_ptr<Modules::Fft> fftSignalConfig =
        std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::Fft> fftFilterConfig =
        std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::Multiply> multiplyConfig =
        std::make_shared<Modules::Multiply>();
    std::shared_ptr<Modules::Fold> foldConfig =
        std::make_shared<Modules::Fold>();
    std::shared_ptr<Modules::Fft> ifftConfig =
        std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::MultiplyConstant> normalizeConfig =
        std::make_shared<Modules::MultiplyConstant>();
    std::shared_ptr<Modules::Unpad> unpadConfig =
        std::make_shared<Modules::Unpad>();
    std::shared_ptr<Modules::OverlapAdd> overlapConfig =
        std::make_shared<Modules::OverlapAdd>();

 private:
    FilterCandidatePlan candidatePlan;
};

Result FilterImpl::validate() {
    const auto& config = *candidate();
    candidatePlan = {};

    if (config.heads == 0) {
        JST_ERROR("[BLOCK_FILTER] Heads must be greater than 0.");
        return Result::ERROR;
    }

    const auto signal = inputs().find("signal");
    if (signal != inputs().end() && signal->second.tensor.validShape()) {
        const Tensor& signalTensor = signal->second.tensor;
        if (signalTensor.rank() == 0) {
            JST_ERROR("[BLOCK_FILTER] Signal input must have at least one dimension.");
            return Result::ERROR;
        }

        const U64 signalSize = signalTensor.shape(signalTensor.rank() - 1);
        if (signalSize == 0) {
            JST_ERROR("[BLOCK_FILTER] Signal input's last dimension cannot be zero.");
            return Result::ERROR;
        }

        // Filter Taps remains responsible for rejecting an invalid tap count.
        if (config.taps != 0) {
            JST_CHECK(CalculateCandidatePlan(config, signalSize, candidatePlan));
        }
    }

    if (heads != config.heads) {
        return Result::RECREATE;
    }

    if (sampleRate != config.sampleRate) {
        return Result::RECREATE;
    }

    if (bandwidth != config.bandwidth) {
        return Result::RECREATE;
    }

    if (taps != config.taps) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result FilterImpl::configure() {
    center.resize(heads);

    filterTapsConfig->sampleRate = sampleRate;
    filterTapsConfig->bandwidth = bandwidth;
    filterTapsConfig->center.resize(center.size());
    for (U64 i = 0; i < center.size(); ++i) {
        filterTapsConfig->center[i] = center[i];
    }
    filterTapsConfig->taps = taps;

    return Result::SUCCESS;
}

Result FilterImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Signal",
                                   "Input signal to filter."));
    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Filtered output signal."));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "The sampling rate of the input signal.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("bandwidth",
                                    "Bandwidth",
                                    "The passband bandwidth of the filter.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("heads",
                                    "Heads",
                                    "Number of filter heads.",
                                    "uint:heads"));

    JST_CHECK(defineInterfaceConfig("center",
                                    "Center",
                                    "The center frequency offset(s) of the filter.",
                                    "vector:float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("taps",
                                    "Taps",
                                    "Number of filter coefficients (must be odd).",
                                    "uint:taps"));

    return Result::SUCCESS;
}

Result FilterImpl::create() {
    const auto& signalPort = inputs().at("signal");
    const Tensor& signalTensor = signalPort.tensor;

    const U64 signalMaxRank = signalTensor.rank() - 1;
    const U64 signalSize = signalTensor.shape(signalMaxRank);

    // Create filter taps.

    JST_CHECK(moduleCreate("filter_taps", filterTapsConfig, {}));

    const Tensor& filterTensor = moduleGetOutput({"filter_taps", "coeffs"}).tensor;
    const U64 filterMaxRank = filterTensor.rank() - 1;
    const U64 filterSize = filterTensor.shape(filterMaxRank);

    // Detect multi-head filter (2D filter tensor).

    const bool multiHead = (filterTensor.rank() == 2);

    // Calculate resampling parameters.

    const bool resample = candidatePlan.resample;
    const U64 padSize = candidatePlan.padSize;
    const U64 resamplerOffset = candidatePlan.resamplerOffset;
    const U64 resamplerSize = candidatePlan.resamplerSize;

    // Expand signal dimensions for multi-head broadcasting.

    auto signalInput = signalPort;

    if (multiHead) {
        expandDimsConfig->axis = signalMaxRank;

        JST_CHECK(moduleCreate("expandDims", expandDimsConfig, {
            {"buffer", signalPort}
        }));

        signalInput = moduleGetOutput({"expandDims", "buffer"});
    }

    const U64 expandedSignalMaxRank =
        multiHead ? signalMaxRank + 1 : signalMaxRank;

    // Pad signal.

    padSignalConfig->size = filterSize - 1;
    padSignalConfig->axis = expandedSignalMaxRank;

    JST_CHECK(moduleCreate("padSignal", padSignalConfig, {
        {"unpadded", signalInput}
    }));

    // Pad filter.

    padFilterConfig->size = signalSize - 1;
    padFilterConfig->axis = filterMaxRank;

    JST_CHECK(moduleCreate("padFilter", padFilterConfig, {
        {"unpadded", moduleGetOutput({"filter_taps", "coeffs"})}
    }));

    // Forward FFT signal.

    fftSignalConfig->forward = true;

    JST_CHECK(moduleCreate("fftSignal", fftSignalConfig, {
        {"signal", moduleGetOutput({"padSignal", "padded"})}
    }));

    // Forward FFT filter.

    fftFilterConfig->forward = true;

    JST_CHECK(moduleCreate("fftFilter", fftFilterConfig, {
        {"signal", moduleGetOutput({"padFilter", "padded"})}
    }));

    // Multiply spectra.

    JST_CHECK(moduleCreate("multiply", multiplyConfig, {
        {"a", moduleGetOutput({"fftSignal", "signal"})},
        {"b", moduleGetOutput({"fftFilter", "signal"})}
    }));

    // Optional fold for resampling.

    auto ifftInput = moduleGetOutput({"multiply", "product"});

    if (resample) {
        const U64 maxRank = multiHead
            ? std::max(filterMaxRank, expandedSignalMaxRank)
            : std::max(filterMaxRank, signalMaxRank);
        foldConfig->axis = maxRank;
        foldConfig->offset = resamplerOffset;
        foldConfig->size = resamplerSize;

        JST_CHECK(moduleCreate("fold", foldConfig, {
            {"buffer", moduleGetOutput({"multiply", "product"})}
        }));

        ifftInput = moduleGetOutput({"fold", "buffer"});
    }

    // Inverse FFT.

    ifftConfig->forward = false;

    JST_CHECK(moduleCreate("ifft", ifftConfig, {
        {"signal", ifftInput}
    }));

    const U64 maxRank = multiHead
        ? std::max(filterMaxRank, expandedSignalMaxRank)
        : std::max(filterMaxRank, signalMaxRank);

    // Normalize the unscaled inverse FFT.

    const auto& ifftOutput = moduleGetOutput({"ifft", "signal"});
    normalizeConfig->constant =
        1.0f / static_cast<F32>(ifftOutput.tensor.shape(maxRank));

    JST_CHECK(moduleCreate("normalize", normalizeConfig, {
        {"factor", ifftOutput}
    }));

    // Unpad.

    unpadConfig->size = padSize;
    unpadConfig->axis = maxRank;

    JST_CHECK(moduleCreate("unpad", unpadConfig, {
        {"padded", moduleGetOutput({"normalize", "product"})}
    }));

    // Overlap-add.

    overlapConfig->axis = maxRank;

    JST_CHECK(moduleCreate("overlap", overlapConfig, {
        {"buffer", moduleGetOutput({"unpad", "unpadded"})},
        {"overlap", moduleGetOutput({"unpad", "pad"})}
    }));

    // Expose output.

    JST_CHECK(moduleExposeOutput("buffer",
                                 {"overlap", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FilterImpl,
                   {"filter_taps"},
                   {"pad"},
                   {"fft"},
                   {"multiply"},
                   {"multiply_constant"},
                   {"unpad"},
                   {"overlap_add"},
                   {"expand_dims", true},
                   {"fold", true});

}  // namespace Jetstream::Blocks
