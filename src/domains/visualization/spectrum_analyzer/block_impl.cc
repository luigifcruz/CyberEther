#include <jetstream/domains/visualization/spectrum_analyzer/block.hh>

#include <memory>
#include <optional>
#include <string>

#include <jetstream/detail/block_impl.hh>
#include <jetstream/domains/dsp/invert/module.hh>
#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/domains/core/range/module.hh>
#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/dsp/amplitude/module.hh>
#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/domains/dsp/window/module.hh>
#include <jetstream/domains/visualization/signal_view/module.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Blocks {

struct SpectrumAnalyzerImpl : public Block::Impl,
                              public DynamicConfig<Blocks::SpectrumAnalyzer> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 private:
    struct CandidatePlan {
        Index sampleAxis = 0;
        U64 windowSize = 0;
        std::string windowShape;
    };

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
    std::shared_ptr<Modules::Amplitude> amplitudeConfig =
        std::make_shared<Modules::Amplitude>();
    std::shared_ptr<Modules::Range> rangeConfig =
        std::make_shared<Modules::Range>();
    std::shared_ptr<Modules::SignalView> signalViewConfig =
        std::make_shared<Modules::SignalView>();

    std::optional<CandidatePlan> candidatePlan;
};

Result SpectrumAnalyzerImpl::validate() {
    candidatePlan.reset();

    const auto input = inputs().find("buffer");
    if (input == inputs().end() || !input->second.resolved()) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = input->second.tensor;
    if (inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[BLOCK_SPECTRUM_ANALYZER] Input must have data type CF32.");
        return Result::ERROR;
    }

    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[BLOCK_SPECTRUM_ANALYZER] Input signal axis metadata is "
                  "invalid.");
        return Result::ERROR;
    }

    if (axes.channel) {
        JST_ERROR("[BLOCK_SPECTRUM_ANALYZER] Channel inputs are not supported.");
        return Result::ERROR;
    }

    for (Index axis = 0; axis < inputTensor.rank(); ++axis) {
        if (axis != *axes.sample && (!axes.batch || axis != *axes.batch)) {
            JST_ERROR("[BLOCK_SPECTRUM_ANALYZER] Unsupported auxiliary input "
                      "axis {}.",
                      axis);
            return Result::ERROR;
        }
    }

    CandidatePlan plan;
    plan.sampleAxis = *axes.sample;
    plan.windowSize = inputTensor.shape(plan.sampleAxis);
    plan.windowShape = "[";
    for (Index dimension = 0; dimension < inputTensor.rank(); ++dimension) {
        if (dimension > 0) {
            plan.windowShape += ", ";
        }
        plan.windowShape += std::to_string(
            dimension == plan.sampleAxis ? plan.windowSize : 1);
    }
    plan.windowShape += "]";
    candidatePlan = std::move(plan);

    return Result::SUCCESS;
}

Result SpectrumAnalyzerImpl::configure() {
    fftConfig->forward = true;
    rangeConfig->min = rangeMin;
    rangeConfig->max = rangeMax;

    signalViewConfig->mode = "lineplot_waterfall";
    signalViewConfig->averaging = averaging;
    signalViewConfig->decimation = decimation;
    signalViewConfig->maxHold = maxHold;
    signalViewConfig->fill = fill;
    signalViewConfig->rangeMin = rangeMin;
    signalViewConfig->rangeMax = rangeMax;
    signalViewConfig->waterfallHeight = waterfallHeight;
    signalViewConfig->xLabel = xLabel;
    signalViewConfig->amplitudeLabel = amplitudeLabel;
    signalViewConfig->waterfallLabel = waterfallLabel;

    return Result::SUCCESS;
}

Result SpectrumAnalyzerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input",
                                   "Complex samples to analyze."));

    JST_CHECK(defineInterfaceConfig("rangeMin", "Range Min",
                                    "Minimum displayed amplitude.",
                                    "range:-300:0:dBFS:float"));
    JST_CHECK(defineInterfaceConfig("rangeMax", "Range Max",
                                    "Maximum displayed amplitude.",
                                    "range:-300:0:dBFS:float"));
    JST_CHECK(defineInterfaceConfig("averaging", "Averaging",
                                    "Trace smoothing factor.",
                                    "range:1:256:samples:uint"));
    JST_CHECK(defineInterfaceConfig("decimation", "Decimation",
                                    "Trace-only horizontal decimation.",
                                    "uint:"));
    JST_CHECK(defineInterfaceConfig("maxHold", "Max Hold",
                                    "Enable maximum hold trace.",
                                    "bool"));
    JST_CHECK(defineInterfaceConfig("waterfallHeight", "Waterfall Height",
                                    "Number of spectrum rows retained.",
                                    "uint:rows"));
    return Result::SUCCESS;
}

Result SpectrumAnalyzerImpl::create() {
    if (!candidatePlan) {
        JST_ERROR("[BLOCK_SPECTRUM_ANALYZER] Input validation plan is "
                  "unavailable.");
        return Result::ERROR;
    }

    windowConfig->size = candidatePlan->windowSize;
    reshapeWindowConfig->shape = candidatePlan->windowShape;

    JST_CHECK(moduleCreate("window", windowConfig, {}));
    auto windowOutput = moduleGetOutput({"window", "window"});
    JST_CHECK(SetSignalAxes(windowOutput.tensor, {
        .sample = Index{0},
    }));

    JST_CHECK(moduleCreate("invert", invertConfig, {
        {"signal", windowOutput}
    }));

    JST_CHECK(moduleCreate("reshape_window", reshapeWindowConfig, {
        {"buffer", moduleGetOutput({"invert", "signal"})}
    }));
    auto reshapedWindow = moduleGetOutput({"reshape_window", "buffer"});
    JST_CHECK(SetSignalAxes(reshapedWindow.tensor, {
        .sample = candidatePlan->sampleAxis,
    }));

    JST_CHECK(moduleCreate("multiply", multiplyConfig, {
        {"a", inputs().at("buffer")},
        {"b", reshapedWindow}
    }));

    JST_CHECK(moduleCreate("fft", fftConfig, {
        {"signal", moduleGetOutput({"multiply", "product"})}
    }));

    JST_CHECK(moduleCreate("amplitude", amplitudeConfig, {
        {"signal", moduleGetOutput({"fft", "signal"})}
    }));

    JST_CHECK(moduleCreate("range", rangeConfig, {
        {"signal", moduleGetOutput({"amplitude", "signal"})}
    }));

    JST_CHECK(moduleCreate("signal_view", signalViewConfig, {
        {"signal", moduleGetOutput({"range", "signal"})}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SpectrumAnalyzerImpl,
                   {"window"},
                   {"invert"},
                   {"reshape"},
                   {"multiply"},
                   {"fft"},
                   {"amplitude"},
                   {"range"},
                   {"signal_view"});

}  // namespace Jetstream::Blocks
