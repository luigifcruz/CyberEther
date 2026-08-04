#include <algorithm>
#include <any>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <string>

#include <jetstream/domains/dsp/filter_engine/block.hh>
#include <jetstream/detail/block_impl.hh>

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
#include <jetstream/domains/dsp/phase_correction/module.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Blocks {

namespace {

struct FilterEngineCandidatePlan {
    bool resample = false;
    Index sampleAxis = 0;
    Index filterSampleAxis = 0;
    bool multiHead = false;
    SignalAxes outputAxes;
    U64 convolutionSize = 0;
    U64 padSize = 0;
    std::vector<U64> resamplerOffsets;
    U64 resamplerSize = 0;
    F32 resampledSampleRate = 0.0f;
};

Result CalculateResampleHeuristics(const std::optional<F32>& sampleRateAttribute,
                                    const std::optional<F32>& bandwidthAttribute,
                                    const std::optional<std::vector<F32>>& centerAttribute,
                                    U64 combinedSize,
                                    FilterEngineCandidatePlan& plan) {
    // Check if filter has all necessary attributes.

    if (!sampleRateAttribute || !bandwidthAttribute || !centerAttribute) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter is not passing necessary attributes.");
        return Result::SUCCESS;
    }

    const F32 sampleRate = *sampleRateAttribute;
    const F32 bandwidth = *bandwidthAttribute;
    const auto& centers = *centerAttribute;

    if (sampleRate <= 0.0f || bandwidth <= 0.0f) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "sampleRate ({}) or bandwidth ({}) is invalid.",
                 sampleRate, bandwidth);
        return Result::SUCCESS;
    }

    const F32 resamplerRatio = sampleRate / bandwidth;

    if (!std::isfinite(resamplerRatio) || resamplerRatio <= 0.0f) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "resampler ratio ({}) is invalid.",
                 resamplerRatio);
        return Result::SUCCESS;
    }

    const F64 u64UpperBound =
        std::ldexp(1.0, std::numeric_limits<U64>::digits);
    if (resamplerRatio != std::floor(resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter bandwidth ({:.2f} MHz) is not a multiple "
                 "of the signal sample rate ({:.2f} MHz).",
                 bandwidth / 1e6f, sampleRate / 1e6f);
        return Result::SUCCESS;
    }

    if (static_cast<F64>(resamplerRatio) >= u64UpperBound) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "resampler ratio ({}) exceeds the supported index range.",
                 resamplerRatio);
        return Result::SUCCESS;
    }

    const U64 integerRatio = static_cast<U64>(resamplerRatio);
    if (plan.padSize % integerRatio != 0) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter tap size minus one ({}) is not a multiple "
                 "of the resampler ratio ({}).",
                 plan.padSize,
                 integerRatio);
        return Result::SUCCESS;
    }

    if (combinedSize % integerRatio != 0) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter tap size minus one ({}) plus signal "
                 "size ({}) is not a multiple of the resampler "
                 "ratio ({}).",
                 plan.padSize, combinedSize - plan.padSize,
                 integerRatio);
        return Result::SUCCESS;
    }

    try {
        plan.resamplerOffsets.assign(centers.size(), 0);
    } catch (const std::exception&) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] Failed to allocate resampler offsets.");
        return Result::ERROR;
    }

    const F32 frequencyPerBin = sampleRate / static_cast<F32>(combinedSize);
    for (U64 head = 0; head < centers.size(); ++head) {
        const F32 center = centers[head];
        if (center == 0.0f) {
            continue;
        }
        const F32 centerBin = center / frequencyPerBin;
        const F64 roundedCenterBin = static_cast<F64>(std::round(centerBin));

        if (!std::isfinite(centerBin) || !std::isfinite(roundedCenterBin) ||
            roundedCenterBin <= -u64UpperBound || roundedCenterBin >= u64UpperBound) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter center ({}) cannot be "
                      "represented as a fold index.", center);
            return Result::ERROR;
        }

        const F64 foldOffsetBin = -roundedCenterBin;
        if (foldOffsetBin < 0.0) {
            const F64 centerBinMagnitude = -foldOffsetBin;
            if (!std::isfinite(centerBinMagnitude) ||
                centerBinMagnitude >= u64UpperBound) {
                JST_ERROR("[BLOCK_FILTER_ENGINE] Filter center ({}) cannot be "
                          "represented as a fold index.", center);
                return Result::ERROR;
            }

            const U64 magnitude = static_cast<U64>(centerBinMagnitude);
            const U64 remainder = magnitude % combinedSize;
            plan.resamplerOffsets[head] =
                remainder == 0 ? 0 : combinedSize - remainder;
        } else {
            plan.resamplerOffsets[head] =
                static_cast<U64>(foldOffsetBin) % combinedSize;
        }

        if (centerBin != roundedCenterBin) {
            JST_WARN("[BLOCK_FILTER_ENGINE] Output will be shifted by "
                     "{} MHz because filter center frequency "
                     "({:.2f} MHz) is not a multiple of the "
                     "frequency per bin ({} MHz).",
                     (centerBin - roundedCenterBin) *
                         frequencyPerBin / 1e6f,
                     center / 1e6f,
                     frequencyPerBin / 1e6f);
        }
    }

    plan.resamplerSize = combinedSize / integerRatio;
    plan.padSize /= integerRatio;
    plan.resampledSampleRate = sampleRate / static_cast<F32>(integerRatio);
    plan.resample = true;

    return Result::SUCCESS;
}

}  // namespace

struct FilterEngineImpl : public Block::Impl,
                           public DynamicConfig<Blocks::FilterEngine> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::optional<FilterEngineCandidatePlan> candidatePlan;
    std::shared_ptr<Modules::Cast> castSignalConfig =
        std::make_shared<Modules::Cast>();
    std::shared_ptr<Modules::Cast> castFilterConfig =
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
    std::shared_ptr<Modules::PhaseCorrection> phaseCorrectionConfig =
        std::make_shared<Modules::PhaseCorrection>();
    std::shared_ptr<Modules::Unpad> unpadConfig =
        std::make_shared<Modules::Unpad>();
    std::shared_ptr<Modules::OverlapAdd> overlapConfig =
        std::make_shared<Modules::OverlapAdd>();
};

Result FilterEngineImpl::validate() {
    candidatePlan.reset();

    const Tensor* signalTensor = nullptr;
    std::optional<SignalAxes> signalAxes;
    const auto signal = inputs().find("signal");
    if (signal != inputs().end() && signal->second.tensor.validShape()) {
        signalTensor = &signal->second.tensor;
        if (signalTensor->dtype() != DataType::F32 &&
            signalTensor->dtype() != DataType::CF32) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Signal input must have data type F32 or CF32.");
            return Result::ERROR;
        }
        SignalAxes axes;
        if (ResolveSignalAxes(*signalTensor, axes) != Result::SUCCESS) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Signal axis metadata is invalid.");
            return Result::ERROR;
        }
        signalAxes = axes;
        if (signalTensor->shape(*signalAxes->sample) == 0) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Signal input's sample dimension "
                      "cannot be zero.");
            return Result::ERROR;
        }
    }

    const Tensor* filterTensor = nullptr;
    std::optional<SignalAxes> filterAxes;
    std::optional<F32> sampleRate;
    std::optional<F32> bandwidth;
    std::optional<std::vector<F32>> centers;
    bool scalarCenter = false;
    const auto filter = inputs().find("filter");
    if (filter != inputs().end() && filter->second.tensor.validShape()) {
        filterTensor = &filter->second.tensor;
        if (filterTensor->dtype() != DataType::F32 &&
            filterTensor->dtype() != DataType::CF32) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter input must have data type F32 or CF32.");
            return Result::ERROR;
        }
        if (filterTensor->rank() < 1 || filterTensor->rank() > 2) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter input must be rank 1 or 2.");
            return Result::ERROR;
        }
        SignalAxes axes;
        if (ResolveSignalAxes(*filterTensor, axes) != Result::SUCCESS ||
            (filterTensor->rank() == 1 &&
             (*axes.sample != 0 || axes.batch || axes.channel)) ||
            (filterTensor->rank() == 2 &&
             (*axes.sample != 1 || axes.batch || axes.channel != Index{0}))) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter coefficients must be [T] with "
                      "sampleAxis=0 or [C,T] with channelAxis=0 and sampleAxis=1.");
            return Result::ERROR;
        }
        filterAxes = axes;
        const Index filterSampleAxis = *axes.sample;
        if (filterTensor->shape(filterSampleAxis) == 0) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter input's sample dimension "
                      "cannot be zero.");
            return Result::ERROR;
        }

        const auto readAttribute = [&](const char* key,
                                       std::optional<F32>& value) -> Result {
            if (!filterTensor->hasAttribute(key)) {
                return Result::SUCCESS;
            }

            const std::any attribute = filterTensor->attribute(key);
            const auto* typedAttribute = std::any_cast<F32>(&attribute);
            if (typedAttribute == nullptr) {
                JST_ERROR("[BLOCK_FILTER_ENGINE] Filter attribute '{}' must be "
                          "F32.", key);
                return Result::ERROR;
            }

            value = *typedAttribute;
            return Result::SUCCESS;
        };

        JST_CHECK(readAttribute("sampleRate", sampleRate));
        JST_CHECK(readAttribute("bandwidth", bandwidth));
        if (filterTensor->hasAttribute("center")) {
            const std::any attribute = filterTensor->attribute("center");
            if (const auto* scalar = std::any_cast<F32>(&attribute)) {
                centers = std::vector<F32>{*scalar};
                scalarCenter = true;
            } else if (const auto* vector =
                           std::any_cast<std::vector<F32>>(&attribute)) {
                if (vector->empty()) {
                    JST_ERROR("[BLOCK_FILTER_ENGINE] Filter attribute 'center' "
                              "cannot be empty.");
                    return Result::ERROR;
                }
                centers = *vector;
            } else {
                JST_ERROR("[BLOCK_FILTER_ENGINE] Filter attribute 'center' must "
                          "be F32 or vector<F32>.");
                return Result::ERROR;
            }
        }
    }

    if (signalTensor != nullptr && filterTensor != nullptr) {
        const bool multiHead = filterAxes->channel.has_value();
        if (multiHead && signalAxes->channel) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Cannot add filter channels to a "
                      "signal that already carries channelAxis.");
            return Result::ERROR;
        }
        const Index signalSampleAxis = *signalAxes->sample;
        const U64 signalSize = signalTensor->shape(signalSampleAxis);
        const Index filterSampleAxis = *filterAxes->sample;
        const U64 filterSize = filterTensor->shape(filterSampleAxis);
        const U64 headCount = multiHead ? filterTensor->shape(0) : 1;
        if (centers) {
            if (scalarCenter && multiHead) {
                try {
                    centers->resize(headCount, centers->front());
                } catch (const std::exception&) {
                    JST_ERROR("[BLOCK_FILTER_ENGINE] Failed to expand scalar "
                              "center metadata across filter channels.");
                    return Result::ERROR;
                }
            } else if (centers->size() != headCount) {
                JST_ERROR("[BLOCK_FILTER_ENGINE] Filter center metadata must match "
                          "the filter channel extent.");
                return Result::ERROR;
            }
        }
        U64 combinedSize = 0;
        if (!detail::CheckedAdd(signalSize, filterSize - 1, combinedSize)) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Combined signal and filter extent "
                      "exceeds the supported range.");
            return Result::ERROR;
        }

        FilterEngineCandidatePlan plan;
        plan.sampleAxis = signalSampleAxis;
        plan.filterSampleAxis = filterSampleAxis;
        plan.multiHead = multiHead;
        plan.convolutionSize = combinedSize;
        plan.outputAxes = *signalAxes;
        if (multiHead) {
            plan.outputAxes.sample = signalSampleAxis + 1;
            plan.outputAxes.channel = signalSampleAxis;
            if (signalAxes->batch) {
                plan.outputAxes.batch = *signalAxes->batch >= signalSampleAxis
                                            ? *signalAxes->batch + 1
                                            : *signalAxes->batch;
            }
        }
        plan.padSize = filterSize - 1;
        JST_CHECK(CalculateResampleHeuristics(sampleRate,
                                              bandwidth,
                                              centers,
                                              combinedSize,
                                              plan));
        candidatePlan = plan;
    }

    return Result::SUCCESS;
}

Result FilterEngineImpl::configure() {
    castSignalConfig->outputType = "CF32";
    castFilterConfig->outputType = "CF32";
    return Result::SUCCESS;
}

Result FilterEngineImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Signal",
                                   "Input signal to filter."));
    JST_CHECK(defineInterfaceInput("filter",
                                   "Filter",
                                   "FIR filter coefficients."));
    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Filtered output signal."));

    return Result::SUCCESS;
}

Result FilterEngineImpl::create() {
    const auto& signalPort = inputs().at("signal");
    const auto& filterPort = inputs().at("filter");
    const Tensor& signalTensor = signalPort.tensor;
    const Tensor& filterTensor = filterPort.tensor;

    if (!candidatePlan) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] Input validation plan is unavailable.");
        return Result::ERROR;
    }

    const Index signalSampleAxis = candidatePlan->sampleAxis;
    const Index filterSampleAxis = candidatePlan->filterSampleAxis;
    const bool multiHead = candidatePlan->multiHead;
    const SignalAxes& outputAxes = candidatePlan->outputAxes;
    Index sampleAxis = signalSampleAxis;
    const U64 signalSize = signalTensor.shape(signalSampleAxis);
    const U64 filterSize = filterTensor.shape(filterSampleAxis);

    // Calculate resampling parameters.

    const bool resample = candidatePlan->resample;
    const U64 padSize = candidatePlan->padSize;
    const auto& resamplerOffsets = candidatePlan->resamplerOffsets;
    const U64 resamplerSize = candidatePlan->resamplerSize;
    const bool applyPhaseCorrection = resample &&
        std::any_of(resamplerOffsets.begin(),
                    resamplerOffsets.end(),
                    [](const U64 offset) { return offset != 0; });

    JST_CHECK(moduleCreate("cast_signal", castSignalConfig, {
        {"buffer", signalPort}
    }));
    auto signalInput = moduleGetOutput({"cast_signal", "buffer"});

    JST_CHECK(moduleCreate("cast_filter", castFilterConfig, {
        {"buffer", filterPort}
    }));
    const auto filterInput = moduleGetOutput({"cast_filter", "buffer"});

    if (signalInput.tensor.dtype() != DataType::CF32 ||
        filterInput.tensor.dtype() != DataType::CF32) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] Internal convolution inputs must be CF32.");
        return Result::ERROR;
    }

    if (multiHead) {
        const Index headAxis = signalSampleAxis;
        expandSignalConfig->axis = static_cast<I64>(headAxis);

        JST_CHECK(moduleCreate("expand_signal", expandSignalConfig, {
            {"buffer", signalInput}
        }));

        signalInput = moduleGetOutput({"expand_signal", "buffer"});
        JST_CHECK(SetSignalAxes(signalInput.tensor, outputAxes));
        ++sampleAxis;
    }

    // Pad signal.

    padSignalConfig->size = filterSize - 1;
    padSignalConfig->axis = static_cast<I64>(sampleAxis);

    JST_CHECK(moduleCreate("padSignal", padSignalConfig, {
        {"unpadded", signalInput}
    }));

    // Pad filter.

    padFilterConfig->size = signalSize - 1;
    padFilterConfig->axis = static_cast<I64>(filterSampleAxis);

    JST_CHECK(moduleCreate("padFilter", padFilterConfig, {
        {"unpadded", filterInput}
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

    const auto signalSpectrum = moduleGetOutput({"fftSignal", "signal"});
    auto filterSpectrum = moduleGetOutput({"fftFilter", "signal"});
    if (signalSpectrum.tensor.dtype() != DataType::CF32 ||
        filterSpectrum.tensor.dtype() != DataType::CF32 ||
        signalSpectrum.tensor.shape(sampleAxis) != candidatePlan->convolutionSize ||
        filterSpectrum.tensor.shape(filterSampleAxis) !=
            candidatePlan->convolutionSize) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] The FFT operands must use full-length CF32 spectra.");
        return Result::ERROR;
    }
    Shape filterSpectrumShape(signalInput.tensor.rank(), 1);
    if (multiHead) {
        filterSpectrumShape[signalSampleAxis] = filterTensor.shape(0);
    }
    filterSpectrumShape[sampleAxis] =
        filterSpectrum.tensor.shape(filterSampleAxis);

    if (filterSpectrum.tensor.shape() != filterSpectrumShape) {
        reshapeFilterConfig->shape = "[";
        for (Index dimension = 0; dimension < filterSpectrumShape.size(); ++dimension) {
            if (dimension > 0) {
                reshapeFilterConfig->shape += ", ";
            }
            reshapeFilterConfig->shape +=
                std::to_string(filterSpectrumShape[dimension]);
        }
        reshapeFilterConfig->shape += "]";

        JST_CHECK(moduleCreate("reshape_filter", reshapeFilterConfig, {
            {"buffer", filterSpectrum}
        }));
        filterSpectrum = moduleGetOutput({"reshape_filter", "buffer"});
    }
    SignalAxes alignedFilterAxes;
    alignedFilterAxes.sample = sampleAxis;
    if (multiHead) {
        alignedFilterAxes.channel = signalSampleAxis;
    }
    JST_CHECK(SetSignalAxes(filterSpectrum.tensor, alignedFilterAxes));

    // Multiply spectra.

    JST_CHECK(moduleCreate("multiply", multiplyConfig, {
        {"a", signalSpectrum},
        {"b", filterSpectrum}
    }));
    auto product = moduleGetOutput({"multiply", "product"});
    if (product.tensor.dtype() != DataType::CF32 ||
        product.tensor.shape(sampleAxis) != candidatePlan->convolutionSize) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] Spectral product must remain full-length CF32.");
        return Result::ERROR;
    }
    JST_CHECK(SetSignalAxes(product.tensor, outputAxes));

    // Optional fold for resampling.

    auto ifftInput = product;
    if (resample && multiHead) {
        JST_CHECK(product.tensor.setAttribute(
            "channelOffsets", resamplerOffsets));
    } else {
        JST_CHECK(product.tensor.removeAttribute("channelOffsets"));
    }

    if (resample) {
        foldConfig->offset = multiHead ? 0 : resamplerOffsets.front();
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
    const U64 expectedIfftSize =
        resample ? resamplerSize : candidatePlan->convolutionSize;
    if (ifftOutput.tensor.dtype() != DataType::CF32 ||
        ifftOutput.tensor.shape(sampleAxis) != expectedIfftSize) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] Inverse FFT must consume the complete CF32 spectrum.");
        return Result::ERROR;
    }
    normalizeConfig->constant =
        1.0f / static_cast<F32>(ifftOutput.tensor.shape(sampleAxis));

    JST_CHECK(moduleCreate("normalize", normalizeConfig, {
        {"factor", ifftOutput}
    }));

    auto normalizedOutput = moduleGetOutput({"normalize", "product"});
    if (!applyPhaseCorrection || !multiHead) {
        JST_CHECK(normalizedOutput.tensor.removeAttribute(
            "channelPhaseIncrements"));
    }
    if (applyPhaseCorrection) {
        if (multiHead) {
            std::vector<F64> channelPhaseIncrements(resamplerOffsets.size());
            for (U64 head = 0; head < resamplerOffsets.size(); ++head) {
                channelPhaseIncrements[head] = std::remainder(
                    2.0 * JST_PI * static_cast<F64>(resamplerOffsets[head]) *
                        static_cast<F64>(signalSize) /
                        static_cast<F64>(candidatePlan->convolutionSize),
                    2.0 * JST_PI);
            }
            JST_CHECK(normalizedOutput.tensor.setAttribute(
                "channelPhaseIncrements", channelPhaseIncrements));
            phaseCorrectionConfig->phaseIncrement = 0.0;
        } else {
            phaseCorrectionConfig->phaseIncrement = std::remainder(
                2.0 * JST_PI * static_cast<F64>(resamplerOffsets.front()) *
                    static_cast<F64>(signalSize) /
                    static_cast<F64>(candidatePlan->convolutionSize),
                2.0 * JST_PI);
        }
        JST_CHECK(moduleCreate("phase_correction", phaseCorrectionConfig, {
            {"signal", normalizedOutput}
        }));
        normalizedOutput = moduleGetOutput({"phase_correction", "signal"});
    }

    if (padSize == 0) {
        if (applyPhaseCorrection) {
            JST_CHECK(moduleExposeOutput("buffer", {"phase_correction", "signal"}));
        } else {
            JST_CHECK(moduleExposeOutput("buffer", {"normalize", "product"}));
        }
    } else {
        // Unpad.

        unpadConfig->size = padSize;
        unpadConfig->axis = static_cast<I64>(sampleAxis);

        JST_CHECK(moduleCreate("unpad", unpadConfig, {
            {"padded", normalizedOutput}
        }));

        // Overlap-add.

        JST_CHECK(moduleCreate("overlap", overlapConfig, {
            {"buffer", moduleGetOutput({"unpad", "unpadded"})},
            {"overlap", moduleGetOutput({"unpad", "pad"})}
        }));

        JST_CHECK(moduleExposeOutput("buffer", {"overlap", "buffer"}));
    }
    JST_CHECK(SetSignalAxes(outputs().at("buffer").tensor, outputAxes));
    if (resample) {
        JST_CHECK(outputs().at("buffer").tensor.setAttribute(
            "sampleRate", candidatePlan->resampledSampleRate));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FilterEngineImpl,
                   {"cast"},
                   {"expand_dims", true},
                   {"pad"},
                   {"fft"},
                   {"reshape", true},
                   {"multiply"},
                   {"multiply_constant"},
                   {"phase_correction", true},
                   {"unpad", true},
                   {"overlap_add", true},
                   {"fold", true});

}  // namespace Jetstream::Blocks
