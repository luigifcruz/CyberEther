#include <any>
#include <cmath>
#include <limits>
#include <optional>

#include <jetstream/domains/dsp/filter_engine/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/expand_dims/module.hh>
#include <jetstream/domains/core/pad/module.hh>
#include <jetstream/domains/core/unpad/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/core/multiply_constant/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/fold/module.hh>
#include <jetstream/domains/dsp/overlap_add/module.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Blocks {

namespace {

struct FilterEngineCandidatePlan {
    bool resample = false;
    U64 padSize = 0;
    U64 resamplerOffset = 0;
    U64 resamplerSize = 0;
};

Result CalculateResampleHeuristics(const std::optional<F32>& sampleRateAttribute,
                                    const std::optional<F32>& bandwidthAttribute,
                                    const std::optional<F32>& centerAttribute,
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
    const F32 center = *centerAttribute;

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

    if (center != 0.0f) {
        const F32 frequencyPerBin =
            sampleRate / static_cast<F32>(combinedSize);
        const F32 centerBin = center / frequencyPerBin;
        const F64 roundedCenterBin = static_cast<F64>(std::round(centerBin));

        if (!std::isfinite(centerBin) || !std::isfinite(roundedCenterBin) ||
            roundedCenterBin <= -u64UpperBound || roundedCenterBin >= u64UpperBound) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter center ({}) cannot be "
                      "represented as a fold index.", center);
            return Result::ERROR;
        }

        if (roundedCenterBin < 0.0) {
            const F64 centerBinMagnitude = -roundedCenterBin;
            if (!std::isfinite(centerBinMagnitude) ||
                centerBinMagnitude >= u64UpperBound) {
                JST_ERROR("[BLOCK_FILTER_ENGINE] Filter center ({}) cannot be "
                          "represented as a fold index.", center);
                return Result::ERROR;
            }

            const U64 magnitude = static_cast<U64>(centerBinMagnitude);
            const U64 remainder = magnitude % combinedSize;
            plan.resamplerOffset =
                remainder == 0 ? 0 : combinedSize - remainder;
        } else {
            plan.resamplerOffset = static_cast<U64>(roundedCenterBin);
        }

        if (centerBin != std::floor(centerBin)) {
            JST_WARN("[BLOCK_FILTER_ENGINE] Output will be shifted by "
                     "{} MHz because filter center frequency "
                     "({:.2f} MHz) is not a multiple of the "
                     "frequency per bin ({} MHz).",
                     (centerBin - std::floor(centerBin)) *
                         frequencyPerBin / 1e6f,
                     center / 1e6f,
                     frequencyPerBin / 1e6f);
        }
    }

    plan.resamplerSize = combinedSize / integerRatio;
    plan.padSize /= integerRatio;
    plan.resample = true;

    return Result::SUCCESS;
}

}  // namespace

struct FilterEngineImpl : public Block::Impl,
                           public DynamicConfig<Blocks::FilterEngine> {
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    std::optional<FilterEngineCandidatePlan> candidatePlan;
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
};

Result FilterEngineImpl::validate() {
    candidatePlan.reset();

    const Tensor* signalTensor = nullptr;
    const auto signal = inputs().find("signal");
    if (signal != inputs().end() && signal->second.tensor.validShape()) {
        signalTensor = &signal->second.tensor;
        if (signalTensor->rank() == 0) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Signal input must have at least "
                      "one dimension.");
            return Result::ERROR;
        }
        if (signalTensor->shape(signalTensor->rank() - 1) == 0) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Signal input's last dimension "
                      "cannot be zero.");
            return Result::ERROR;
        }
    }

    const Tensor* filterTensor = nullptr;
    std::optional<F32> sampleRate;
    std::optional<F32> bandwidth;
    std::optional<F32> center;
    const auto filter = inputs().find("filter");
    if (filter != inputs().end() && filter->second.tensor.validShape()) {
        filterTensor = &filter->second.tensor;
        if (filterTensor->rank() == 0) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter input must have at least "
                      "one dimension.");
            return Result::ERROR;
        }
        if (filterTensor->shape(filterTensor->rank() - 1) == 0) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Filter input's last dimension "
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
        JST_CHECK(readAttribute("center", center));
    }

    if (signalTensor != nullptr && filterTensor != nullptr) {
        const U64 signalSize = signalTensor->shape(signalTensor->rank() - 1);
        const U64 filterSize = filterTensor->shape(filterTensor->rank() - 1);
        U64 combinedSize = 0;
        if (!detail::CheckedAdd(signalSize, filterSize - 1, combinedSize)) {
            JST_ERROR("[BLOCK_FILTER_ENGINE] Combined signal and filter extent "
                      "exceeds the supported range.");
            return Result::ERROR;
        }

        FilterEngineCandidatePlan plan;
        plan.padSize = filterSize - 1;
        JST_CHECK(CalculateResampleHeuristics(sampleRate,
                                              bandwidth,
                                              center,
                                              combinedSize,
                                              plan));
        candidatePlan = plan;
    }

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

    const U64 signalMaxRank = signalTensor.rank() - 1;
    const U64 filterMaxRank = filterTensor.rank() - 1;

    const U64 signalSize = signalTensor.shape(signalMaxRank);
    const U64 filterSize = filterTensor.shape(filterMaxRank);

    // Detect multi-head filter (2D filter tensor).

    const bool multiHead = (filterTensor.rank() == 2);

    // Calculate resampling parameters.

    const bool resample = candidatePlan->resample;
    const U64 padSize = candidatePlan->padSize;
    const U64 resamplerOffset = candidatePlan->resamplerOffset;
    const U64 resamplerSize = candidatePlan->resamplerSize;

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

JST_REGISTER_BLOCK(FilterEngineImpl,
                   {"pad"},
                   {"fft"},
                   {"multiply"},
                   {"multiply_constant"},
                   {"unpad"},
                   {"overlap_add"},
                   {"expand_dims", true},
                   {"fold", true});

}  // namespace Jetstream::Blocks
