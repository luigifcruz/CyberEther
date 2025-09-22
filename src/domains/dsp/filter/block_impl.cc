#include <cmath>

#include <jetstream/domains/dsp/filter/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/filter_taps/module.hh>
#include <jetstream/domains/core/expand_dims/module.hh>
#include <jetstream/domains/core/pad/module.hh>
#include <jetstream/domains/core/unpad/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/fold/module.hh>
#include <jetstream/domains/dsp/overlap_add/module.hh>
#include <jetstream/domains/core/duplicate/module.hh>

namespace Jetstream::Blocks {

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
    std::shared_ptr<Modules::Unpad> unpadConfig =
        std::make_shared<Modules::Unpad>();
    std::shared_ptr<Modules::OverlapAdd> overlapConfig =
        std::make_shared<Modules::OverlapAdd>();

 private:
    bool calculateResampleHeuristics(U64 filterSize,
                                     U64 signalSize,
                                     U64& padSize,
                                     U64& resamplerOffset,
                                     U64& resamplerSize);
};

Result FilterImpl::validate() {
    const auto& config = *candidate();

    if (config.heads == 0) {
        JST_ERROR("[BLOCK_FILTER] Heads must be greater than 0.");
        return Result::ERROR;
    }

    if (config.center.empty()) {
        JST_ERROR("[BLOCK_FILTER] At least one center frequency is required.");
        return Result::ERROR;
    }

    if (config.heads < config.center.size()) {
        JST_ERROR("[BLOCK_FILTER] Heads ({}) cannot be lower than the number "
                  "of center entries ({}). Reduce center entries first to "
                  "avoid data loss.",
                  config.heads, config.center.size());
        return Result::ERROR;
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
    if (center.size() < heads) {
        center.resize(heads);
    }

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
                                    "int:heads"));

    JST_CHECK(defineInterfaceConfig("center",
                                    "Center",
                                    "The center frequency offset(s) of the filter.",
                                    "vector:float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("taps",
                                    "Taps",
                                    "Number of filter coefficients (must be odd).",
                                    "int:taps"));

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

    U64 padSize = filterSize - 1;
    U64 resamplerOffset = 0;
    U64 resamplerSize = 0;

    const bool resample = calculateResampleHeuristics(filterSize,
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

bool FilterImpl::calculateResampleHeuristics(U64 filterSize,
                                             U64 signalSize,
                                             U64& padSize,
                                             U64& resamplerOffset,
                                             U64& resamplerSize) {
    const F64 sr = filterTapsConfig->sampleRate;
    const F64 bw = filterTapsConfig->bandwidth;
    const F64 ct = filterTapsConfig->center[0];

    const F64 resamplerRatio = sr / bw;

    if (resamplerRatio != std::floor(resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER] Bypassing resampling because "
                 "filter bandwidth ({:.2f} MHz) is not a multiple "
                 "of the signal sample rate ({:.2f} MHz).",
                 bw / 1e6, sr / 1e6);
        return false;
    }

    if (static_cast<F64>(padSize) /
        resamplerRatio != std::floor(
        static_cast<F64>(padSize) / resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER] Bypassing resampling because "
                 "filter tap size minus one ({}) is not a multiple "
                 "of the resampler ratio ({}).",
                 padSize,
                 static_cast<U64>(resamplerRatio));
        return false;
    }

    resamplerSize = filterSize + signalSize - 1;

    if (static_cast<F64>(resamplerSize) /
        resamplerRatio != std::floor(
        static_cast<F64>(resamplerSize) / resamplerRatio)) {
        JST_WARN("[BLOCK_FILTER] Bypassing resampling because "
                 "filter tap size minus one ({}) plus signal "
                 "size ({}) is not a multiple of the resampler "
                 "ratio ({}).",
                 padSize, signalSize,
                 static_cast<U64>(resamplerRatio));
        return false;
    }

    if (ct != 0.0) {
        const F64 frequencyPerBin =
            sr / static_cast<F64>(resamplerSize);
        const F64 centerBin = ct / frequencyPerBin;

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

        resamplerOffset =
            static_cast<U64>(std::round(centerBin));
    }

    resamplerSize /= static_cast<U64>(resamplerRatio);
    padSize /= static_cast<U64>(resamplerRatio);

    return true;
}

JST_REGISTER_BLOCK(FilterImpl);

}  // namespace Jetstream::Blocks
