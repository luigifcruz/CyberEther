#include <cmath>
#include <limits>
#include <optional>
#include <string>

#include <jetstream/domains/dsp/filter/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <jetstream/tools/numeric.hh>

#include <jetstream/domains/dsp/filter_taps/module.hh>
#include <jetstream/domains/core/cast/module.hh>
#include <jetstream/domains/core/expand_dims/module.hh>
#include <jetstream/domains/core/pad/module.hh>
#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/core/unpad/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/core/multiply_constant/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/fold/module.hh>
#include <jetstream/domains/dsp/overlap_add/module.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Blocks {

namespace {

struct FilterCandidatePlan {
    bool resample = false;
    SignalAxes outputAxes;
    U64 padSize = 0;
    U64 resamplerOffset = 0;
    U64 resamplerSize = 0;
    F32 resampledSampleRate = 0.0f;
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
    candidatePlan.resampledSampleRate = static_cast<F32>(
        sr / static_cast<F64>(resamplerRatioInteger));
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
    std::shared_ptr<Modules::Cast> castSignalConfig =
        std::make_shared<Modules::Cast>();
    std::shared_ptr<Modules::ExpandDims> expandSignalConfig =
        std::make_shared<Modules::ExpandDims>();
    std::shared_ptr<Modules::Pad> padSignalConfig =
        std::make_shared<Modules::Pad>();
    std::shared_ptr<Modules::Pad> padFilterConfig =
        std::make_shared<Modules::Pad>();
    std::shared_ptr<Modules::Fft> fftSignalConfig =
        std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::Fft> fftFilterConfig =
        std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::Reshape> reshapeFilterConfig =
        std::make_shared<Modules::Reshape>();
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
    std::optional<SignalAxes> candidateSignalAxes;
};

Result FilterImpl::validate() {
    const auto& config = *candidate();
    candidatePlan = {};
    candidateSignalAxes.reset();

    if (config.heads == 0) {
        JST_ERROR("[BLOCK_FILTER] Heads must be greater than 0.");
        return Result::ERROR;
    }

    const auto signal = inputs().find("signal");
    if (signal != inputs().end() && signal->second.tensor.validShape()) {
        const Tensor& signalTensor = signal->second.tensor;
        if (signalTensor.dtype() != DataType::F32 &&
            signalTensor.dtype() != DataType::CF32) {
            JST_ERROR("[BLOCK_FILTER] Signal input must have data type F32 or CF32.");
            return Result::ERROR;
        }
        SignalAxes axes;
        if (ResolveSignalAxes(signalTensor, axes) != Result::SUCCESS) {
            JST_ERROR("[BLOCK_FILTER] Signal axis metadata is invalid.");
            return Result::ERROR;
        }
        if (axes.channel) {
            JST_ERROR("[BLOCK_FILTER] Signal already has channelAxis. Generated "
                      "filter channels cannot be nested.");
            return Result::ERROR;
        }
        candidateSignalAxes = axes;

        const Index signalSampleAxis = *axes.sample;
        const U64 signalSize = signalTensor.shape(signalSampleAxis);
        if (signalSize == 0) {
            JST_ERROR("[BLOCK_FILTER] Signal input's sample dimension cannot be zero.");
            return Result::ERROR;
        }

        // Filter Taps remains responsible for rejecting an invalid tap count.
        if (config.taps != 0) {
            JST_CHECK(CalculateCandidatePlan(config, signalSize, candidatePlan));
        }

        candidatePlan.outputAxes.sample = signalSampleAxis + 1;
        candidatePlan.outputAxes.channel = signalSampleAxis;
        if (axes.batch) {
            candidatePlan.outputAxes.batch = *axes.batch >= signalSampleAxis
                                                 ? *axes.batch + 1
                                                 : *axes.batch;
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

    castSignalConfig->outputType = "CF32";

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

    if (!candidateSignalAxes) {
        JST_ERROR("[BLOCK_FILTER] Input validation plan is unavailable.");
        return Result::ERROR;
    }

    const Index signalSampleAxis = *candidateSignalAxes->sample;
    const Index headAxis = signalSampleAxis;
    const Index sampleAxis = signalSampleAxis + 1;
    const SignalAxes& outputAxes = candidatePlan.outputAxes;
    const U64 signalSize = signalTensor.shape(signalSampleAxis);

    // Create filter taps.

    JST_CHECK(moduleCreate("filter_taps", filterTapsConfig, {}));

    auto filterPort = moduleGetOutput({"filter_taps", "coeffs"});
    JST_CHECK(SetSignalAxes(filterPort.tensor, {
        .sample = Index{1},
        .channel = Index{0},
    }));
    const Tensor& filterTensor = filterPort.tensor;
    const U64 filterSize = filterTensor.shape(1);

    // Calculate resampling parameters.

    const bool resample = candidatePlan.resample;
    const U64 padSize = candidatePlan.padSize;
    const U64 resamplerOffset = candidatePlan.resamplerOffset;
    const U64 resamplerSize = candidatePlan.resamplerSize;

    // Promote real inputs before entering the complex convolution pipeline.

    JST_CHECK(moduleCreate("cast_signal", castSignalConfig, {
        {"buffer", signalPort}
    }));
    const auto complexSignal = moduleGetOutput({"cast_signal", "buffer"});

    // Insert the generated filter head immediately before the sample axis.

    expandSignalConfig->axis = static_cast<I64>(headAxis);

    JST_CHECK(moduleCreate("expand_signal", expandSignalConfig, {
        {"buffer", complexSignal}
    }));

    auto signalInput = moduleGetOutput({"expand_signal", "buffer"});
    JST_CHECK(SetSignalAxes(signalInput.tensor, outputAxes));

    // Pad signal.

    padSignalConfig->size = filterSize - 1;
    padSignalConfig->axis = static_cast<I64>(sampleAxis);

    JST_CHECK(moduleCreate("padSignal", padSignalConfig, {
        {"unpadded", signalInput}
    }));

    // Pad filter.

    padFilterConfig->size = signalSize - 1;
    padFilterConfig->axis = 1;

    JST_CHECK(moduleCreate("padFilter", padFilterConfig, {
        {"unpadded", filterPort}
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

    const auto filterFftOutput = moduleGetOutput({"fftFilter", "signal"});
    Shape filterSpectrumShape(signalInput.tensor.rank(), 1);
    filterSpectrumShape[headAxis] = filterTensor.shape(0);
    filterSpectrumShape[sampleAxis] = filterFftOutput.tensor.shape(1);

    reshapeFilterConfig->shape = "[";
    for (Index dimension = 0; dimension < filterSpectrumShape.size(); ++dimension) {
        if (dimension > 0) {
            reshapeFilterConfig->shape += ", ";
        }
        reshapeFilterConfig->shape += std::to_string(filterSpectrumShape[dimension]);
    }
    reshapeFilterConfig->shape += "]";

    JST_CHECK(moduleCreate("reshape_filter", reshapeFilterConfig, {
        {"buffer", filterFftOutput}
    }));
    auto alignedFilterSpectrum = moduleGetOutput({"reshape_filter", "buffer"});
    JST_CHECK(SetSignalAxes(alignedFilterSpectrum.tensor, {
        .sample = sampleAxis,
        .channel = headAxis,
    }));

    // Multiply spectra.

    JST_CHECK(moduleCreate("multiply", multiplyConfig, {
        {"a", moduleGetOutput({"fftSignal", "signal"})},
        {"b", alignedFilterSpectrum}
    }));
    auto product = moduleGetOutput({"multiply", "product"});
    JST_CHECK(SetSignalAxes(product.tensor, outputAxes));

    // Optional fold for resampling.

    auto ifftInput = product;

    if (resample) {
        foldConfig->offset = resamplerOffset;
        foldConfig->size = resamplerSize;

        JST_CHECK(moduleCreate("fold", foldConfig, {
            {"buffer", product}
        }));

        ifftInput = moduleGetOutput({"fold", "buffer"});
    }

    // Inverse FFT.

    ifftConfig->forward = false;

    JST_CHECK(moduleCreate("ifft", ifftConfig, {
        {"signal", ifftInput}
    }));

    // Normalize the unscaled inverse FFT.

    const auto& ifftOutput = moduleGetOutput({"ifft", "signal"});
    normalizeConfig->constant =
        1.0f / static_cast<F32>(ifftOutput.tensor.shape(sampleAxis));

    JST_CHECK(moduleCreate("normalize", normalizeConfig, {
        {"factor", ifftOutput}
    }));

    // Unpad.

    unpadConfig->size = padSize;
    unpadConfig->axis = static_cast<I64>(sampleAxis);

    JST_CHECK(moduleCreate("unpad", unpadConfig, {
        {"padded", moduleGetOutput({"normalize", "product"})}
    }));

    // Overlap-add.

    JST_CHECK(moduleCreate("overlap", overlapConfig, {
        {"buffer", moduleGetOutput({"unpad", "unpadded"})},
        {"overlap", moduleGetOutput({"unpad", "pad"})}
    }));

    // Expose output.

    JST_CHECK(moduleExposeOutput("buffer",
                                 {"overlap", "buffer"}));
    JST_CHECK(SetSignalAxes(outputs().at("buffer").tensor, outputAxes));
    if (resample) {
        JST_CHECK(outputs().at("buffer").tensor.setAttribute(
            "sampleRate", candidatePlan.resampledSampleRate));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FilterImpl,
                   {"filter_taps"},
                   {"cast"},
                   {"expand_dims"},
                   {"pad"},
                   {"fft"},
                   {"reshape"},
                   {"multiply"},
                   {"multiply_constant"},
                   {"unpad"},
                   {"overlap_add"},
                   {"fold", true});

}  // namespace Jetstream::Blocks
