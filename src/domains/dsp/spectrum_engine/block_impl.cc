#include <jetstream/domains/dsp/spectrum_engine/block.hh>

#include <optional>

#include <jetstream/detail/block_impl.hh>
#include <jetstream/domains/core/cast/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/core/range/module.hh>
#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/dsp/agc/module.hh>
#include <jetstream/domains/dsp/amplitude/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/invert/module.hh>
#include <jetstream/domains/dsp/window/module.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Blocks {

struct SpectrumEngineImpl : public Block::Impl,
                            public DynamicConfig<Blocks::SpectrumEngine> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Window> windowConfig =
        std::make_shared<Modules::Window>();
    std::shared_ptr<Modules::Cast> castInputConfig =
        std::make_shared<Modules::Cast>();
    std::shared_ptr<Modules::Invert> invertConfig =
        std::make_shared<Modules::Invert>();
    std::shared_ptr<Modules::Reshape> reshapeWindowConfig =
        std::make_shared<Modules::Reshape>();
    std::shared_ptr<Modules::Multiply> multiplyConfig =
        std::make_shared<Modules::Multiply>();
    std::shared_ptr<Modules::Fft> fftConfig =
        std::make_shared<Modules::Fft>();
    std::shared_ptr<Modules::Agc> agcConfig =
        std::make_shared<Modules::Agc>();
    std::shared_ptr<Modules::Amplitude> amplitudeConfig =
        std::make_shared<Modules::Amplitude>();
    std::shared_ptr<Modules::Range> rangeConfig =
        std::make_shared<Modules::Range>();
    std::optional<Index> candidateSampleAxis;
};

Result SpectrumEngineImpl::validate() {
    const auto& config = *candidate();
    candidateSampleAxis.reset();

    const auto input = inputs().find("buffer");
    if (input != inputs().end() && input->second.resolved()) {
        if (input->second.tensor.dtype() != DataType::F32 &&
            input->second.tensor.dtype() != DataType::CF32) {
            JST_ERROR("[BLOCK_SPECTRUM_ENGINE] Input must have data type F32 "
                      "or CF32.");
            return Result::ERROR;
        }
        SignalAxes axes;
        if (ResolveSignalAxes(input->second.tensor, axes) != Result::SUCCESS) {
            JST_ERROR("[BLOCK_SPECTRUM_ENGINE] Input signal axis metadata is "
                      "invalid.");
            return Result::ERROR;
        }
        candidateSampleAxis = *axes.sample;
    }

    if (enableAgc != config.enableAgc) {
        return Result::RECREATE;
    }

    if (enableScale != config.enableScale) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result SpectrumEngineImpl::configure() {
    castInputConfig->outputType = "CF32";
    fftConfig->forward = true;
    rangeConfig->min = rangeMin;
    rangeConfig->max = rangeMax;

    return Result::SUCCESS;
}

Result SpectrumEngineImpl::define() {
    const auto& config = *candidate();

    JST_CHECK(defineInterfaceInput("buffer", "Input",
                                   "Input signal to compute the spectrum of."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output",
                                    "Spectrum output in decibels."));

    JST_CHECK(defineInterfaceConfig("enableAgc",
                                    "Enable AGC",
                                    "Apply automatic gain control after FFT.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("enableScale",
                                    "Enable Scale",
                                    "Apply range scaling to the output.",
                                    "bool"));

    if (config.enableScale) {
        JST_CHECK(defineInterfaceConfig("rangeMin",
                                        "Range Min",
                                        "Minimum value of the scale range.",
                                        "range:-300:0:dBFS:float"));

        JST_CHECK(defineInterfaceConfig("rangeMax",
                                        "Range Max",
                                        "Maximum value of the scale range.",
                                        "range:-300:0:dBFS:float"));
    }

    return Result::SUCCESS;
}

Result SpectrumEngineImpl::create() {
    const auto& inputPort = inputs().at("buffer");
    const Tensor& inputTensor = inputPort.tensor;

    if (!candidateSampleAxis) {
        JST_ERROR("[BLOCK_SPECTRUM_ENGINE] Input validation plan is unavailable.");
        return Result::ERROR;
    }
    const Index resolvedAxis = *candidateSampleAxis;

    JST_CHECK(moduleCreate("cast_input", castInputConfig, {
        {"buffer", inputPort}
    }));
    const auto complexInput = moduleGetOutput({"cast_input", "buffer"});

    // Derive window size from input shape at specified axis.

    windowConfig->size = inputTensor.shape(resolvedAxis);

    std::string windowShape = "[";
    for (Index dimension = 0; dimension < inputTensor.rank(); ++dimension) {
        if (dimension > 0) {
            windowShape += ", ";
        }
        windowShape += std::to_string(
            dimension == resolvedAxis ? windowConfig->size : 1);
    }
    windowShape += "]";
    reshapeWindowConfig->shape = windowShape;

    // Create window coefficients.

    JST_CHECK(moduleCreate("window", windowConfig, {}));
    auto windowOutput = moduleGetOutput({"window", "window"});
    JST_CHECK(SetSignalAxes(windowOutput.tensor, {
        .sample = Index{0},
    }));

    // Invert window (FFT shift).

    JST_CHECK(moduleCreate("invert", invertConfig, {
        {"signal", windowOutput}
    }));

    // Align the 1D window with the selected input axis for broadcasting.

    JST_CHECK(moduleCreate("reshape_window", reshapeWindowConfig, {
        {"buffer", moduleGetOutput({"invert", "signal"})}
    }));
    auto reshapedWindow = moduleGetOutput({"reshape_window", "buffer"});
    JST_CHECK(SetSignalAxes(reshapedWindow.tensor, {
        .sample = resolvedAxis,
    }));

    // Multiply input signal by shifted window.

    JST_CHECK(moduleCreate("multiply", multiplyConfig, {
        {"a", complexInput},
        {"b", reshapedWindow}
    }));

    // Forward FFT.

    JST_CHECK(moduleCreate("fft", fftConfig, {
        {"signal", moduleGetOutput({"multiply", "product"})}
    }));

    // Optional AGC.

    if (enableAgc) {
        // Preserve relative FFT-bin levels by using one RMS tile per spectrum.
        agcConfig->tileSize = inputTensor.shape(resolvedAxis);
        JST_CHECK(moduleCreate("agc", agcConfig, {
            {"signal", moduleGetOutput({"fft", "signal"})}
        }));

        JST_CHECK(moduleCreate("amplitude", amplitudeConfig, {
            {"signal", moduleGetOutput({"agc", "signal"})}
        }));
    } else {
        JST_CHECK(moduleCreate("amplitude", amplitudeConfig, {
            {"signal", moduleGetOutput({"fft", "signal"})}
        }));
    }

    // Optional Range scaling.

    if (enableScale) {
        JST_CHECK(moduleCreate("range", rangeConfig, {
            {"signal", moduleGetOutput({"amplitude", "signal"})}
        }));

        JST_CHECK(moduleExposeOutput("buffer", {"range", "signal"}));
    } else {
        JST_CHECK(moduleExposeOutput("buffer", {"amplitude", "signal"}));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SpectrumEngineImpl,
                   {"cast"},
                   {"window"},
                   {"invert"},
                   {"reshape"},
                   {"multiply"},
                   {"fft"},
                   {"amplitude"},
                   {"agc", true},
                   {"range", true});

}  // namespace Jetstream::Blocks
