#include <any>
#include <cmath>

#include <jetstream/domains/dsp/filter_engine/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/expand_dims/module.hh>
#include <jetstream/domains/core/pad/module.hh>
#include <jetstream/domains/core/unpad/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/fold/module.hh>
#include <jetstream/domains/dsp/overlap_add/module.hh>

namespace Jetstream::Blocks {

struct FilterEngineImpl : public Block::Impl,
                          public DynamicConfig<Blocks::FilterEngine> {
    Result define() override;
    Result create() override;

 protected:
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
    std::shared_ptr<Modules::Unpad> unpadConfig =
        std::make_shared<Modules::Unpad>();
    std::shared_ptr<Modules::OverlapAdd> overlapConfig =
        std::make_shared<Modules::OverlapAdd>();

 private:
    bool calculateResampleHeuristics(const Tensor& filterTensor,
                                     U64 filterSize,
                                     U64 signalSize,
                                     U64& padSize,
                                     U64& resamplerOffset,
                                     U64& resamplerSize);
};

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

    if (signalTensor.rank() == 0) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] Signal input must have at least "
                  "one dimension.");
        return Result::ERROR;
    }

    if (filterTensor.rank() == 0) {
        JST_ERROR("[BLOCK_FILTER_ENGINE] Filter input must have at least "
                  "one dimension.");
        return Result::ERROR;
    }

    const U64 signalMaxRank = signalTensor.rank() - 1;
    const U64 filterMaxRank = filterTensor.rank() - 1;

    const U64 signalSize = signalTensor.shape(signalMaxRank);
    const U64 filterSize = filterTensor.shape(filterMaxRank);

    // Detect multi-head filter (2D filter tensor).

    const bool multiHead = (filterTensor.rank() == 2);

    // Calculate resampling parameters.

    U64 padSize = filterSize - 1;
    U64 resamplerOffset = 0;
    U64 resamplerSize = 0;

    const bool resample = calculateResampleHeuristics(filterTensor,
                                                      filterSize,
                                                      signalSize,
                                                      padSize,
                                                      resamplerOffset,
                                                      resamplerSize);

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

    // Unpad.

    const U64 maxRank = multiHead
        ? std::max(filterMaxRank, expandedSignalMaxRank)
        : std::max(filterMaxRank, signalMaxRank);

    unpadConfig->size = padSize;
    unpadConfig->axis = maxRank;

    JST_CHECK(moduleCreate("unpad", unpadConfig, {
        {"padded", moduleGetOutput({"ifft", "signal"})}
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

bool FilterEngineImpl::calculateResampleHeuristics(const Tensor& filterTensor,
                                                   U64 filterSize,
                                                   U64 signalSize,
                                                   U64& padSize,
                                                   U64& resamplerOffset,
                                                   U64& resamplerSize) {
    // Check if filter has all necessary attributes.

    if (!filterTensor.hasAttribute("sampleRate") ||
        !filterTensor.hasAttribute("bandwidth") ||
        !filterTensor.hasAttribute("center")) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter is not passing necessary attributes.");
        return false;
    }

    const F32 sampleRate = std::any_cast<F32>(filterTensor.attribute("sampleRate"));
    const F32 bandwidth = std::any_cast<F32>(filterTensor.attribute("bandwidth"));
    const F32 center = std::any_cast<F32>(filterTensor.attribute("center"));

    if (sampleRate <= 0.0f || bandwidth <= 0.0f) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "sampleRate ({}) or bandwidth ({}) is invalid.",
                 sampleRate, bandwidth);
        return false;
    }

    const F32 resamplerRatio = sampleRate / bandwidth;

    if (!std::isfinite(resamplerRatio) || resamplerRatio <= 0.0f) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "resampler ratio ({}) is invalid.",
                 resamplerRatio);
        return false;
    }

    if (resamplerRatio != std::floor(resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter bandwidth ({:.2f} MHz) is not a multiple "
                 "of the signal sample rate ({:.2f} MHz).",
                 bandwidth / 1e6f, sampleRate / 1e6f);
        return false;
    }

    if (static_cast<F32>(padSize) /
        resamplerRatio != std::floor(
        static_cast<F32>(padSize) / resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter tap size minus one ({}) is not a multiple "
                 "of the resampler ratio ({}).",
                 padSize,
                 static_cast<U64>(resamplerRatio));
        return false;
    }

    resamplerSize = filterSize + signalSize - 1;

    if (static_cast<F32>(resamplerSize) /
        resamplerRatio != std::floor(
        static_cast<F32>(resamplerSize) / resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER_ENGINE] Bypassing resampling because "
                 "filter tap size minus one ({}) plus signal "
                 "size ({}) is not a multiple of the resampler "
                 "ratio ({}).",
                 padSize, signalSize,
                 static_cast<U64>(resamplerRatio));
        return false;
    }

    if (center != 0.0f) {
        const F32 frequencyPerBin =
            sampleRate / static_cast<F32>(resamplerSize);
        const F32 centerBin = center / frequencyPerBin;

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

        resamplerOffset =
            static_cast<U64>(std::round(centerBin));
    }

    resamplerSize /= static_cast<U64>(resamplerRatio);
    padSize /= static_cast<U64>(resamplerRatio);

    return true;
}

JST_REGISTER_BLOCK(FilterEngineImpl);

}  // namespace Jetstream::Blocks
