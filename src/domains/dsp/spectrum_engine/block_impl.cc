#include <jetstream/domains/dsp/spectrum_engine/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/dsp/window/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/agc/module.hh>
#include <jetstream/domains/dsp/amplitude/module.hh>
#include <jetstream/domains/core/invert/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/core/range/module.hh>
#include <jetstream/domains/core/reshape/module.hh>
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
};

Result SpectrumEngineImpl::validate() {
    const auto& config = *candidate();

    if (enableAgc != config.enableAgc) {
        return Result::RECREATE;
    }

    if (enableScale != config.enableScale) {
        return Result::RECREATE;
    }

    if (axis != config.axis) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result SpectrumEngineImpl::configure() {
    fftConfig->forward = true;
    rangeConfig->min = rangeMin;
    rangeConfig->max = rangeMax;

    return Result::SUCCESS;
}

Result SpectrumEngineImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input",
                                   "Input signal to compute the spectrum of."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output",
                                    "Spectrum output in decibels."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Axis along which to compute the spectrum. Negative axes "
                                    "count from the end.",
                                    "int:"));

    JST_CHECK(defineInterfaceConfig("enableAgc",
                                    "Enable AGC",
                                    "Apply automatic gain control after FFT.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("enableScale",
                                    "Enable Scale",
                                    "Apply range scaling to the output.",
                                    "bool"));

    if (enableScale) {
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

    const auto candidateAxis = ResolveAxis(axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[BLOCK_SPECTRUM_ENGINE] Axis {} is out of bounds for "
                  "input tensor rank {}.", axis, inputTensor.rank());
        return Result::ERROR;
    }
    const Index resolvedAxis = *candidateAxis;

    const I64 childAxis = static_cast<I64>(resolvedAxis);
    fftConfig->axis = childAxis;
    amplitudeConfig->axis = childAxis;

    // Derive window size from input shape at specified axis.

    windowConfig->size = inputTensor.shape(resolvedAxis);

    std::string windowShape = "[";
    for (Index dimension = 0; dimension < inputTensor.rank(); ++dimension) {
        if (dimension > 0) {
            windowShape += ", ";
        }
        windowShape += std::to_string(dimension == resolvedAxis ? windowConfig->size : 1);
    }
    windowShape += "]";
    reshapeWindowConfig->shape = windowShape;

    // Create window coefficients.

    JST_CHECK(moduleCreate("window", windowConfig, {}));

    // Invert window (FFT shift).

    JST_CHECK(moduleCreate("invert", invertConfig, {
        {"signal", moduleGetOutput({"window", "window"})}
    }));

    // Align the 1D window with the selected input axis for broadcasting.

    JST_CHECK(moduleCreate("reshape_window", reshapeWindowConfig, {
        {"buffer", moduleGetOutput({"invert", "signal"})}
    }));

    // Multiply input signal by shifted window.

    JST_CHECK(moduleCreate("multiply", multiplyConfig, {
        {"a", inputPort},
        {"b", moduleGetOutput({"reshape_window", "buffer"})}
    }));

    // Forward FFT.

    JST_CHECK(moduleCreate("fft", fftConfig, {
        {"signal", moduleGetOutput({"multiply", "product"})}
    }));

    // Optional AGC.

    if (enableAgc) {
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

JST_REGISTER_BLOCK(SpectrumEngineImpl);

}  // namespace Jetstream::Blocks
