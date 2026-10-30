#include <jetstream/domains/visualization/eye_diagram/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/eye_diagram/module.hh>
#include <jetstream/domains/visualization/signal_view/module.hh>

namespace Jetstream::Blocks {

struct EyeDiagramImpl : public Block::Impl, public DynamicConfig<Blocks::EyeDiagram> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::EyeDiagram> eyeDiagramConfig =
        std::make_shared<Modules::EyeDiagram>();
    std::shared_ptr<Modules::SignalView> signalViewConfig =
        std::make_shared<Modules::SignalView>();
};

Result EyeDiagramImpl::configure() {
    eyeDiagramConfig->samplesPerSymbol = samplesPerSymbol;
    eyeDiagramConfig->interpolation = interpolation;
    eyeDiagramConfig->component = component;
    eyeDiagramConfig->rangeMin = rangeMin;
    eyeDiagramConfig->rangeMax = rangeMax;

    signalViewConfig->mode = "lineplot";
    signalViewConfig->fill = false;
    signalViewConfig->density = true;
    signalViewConfig->persistence = persistence;
    signalViewConfig->rangeMin = rangeMin;
    signalViewConfig->rangeMax = rangeMax;
    signalViewConfig->xLabel = "Time (symbols)";
    signalViewConfig->amplitudeLabel = "Amplitude";

    return Result::SUCCESS;
}

Result EyeDiagramImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Oversampled F32 or CF32 signal."));

    JST_CHECK(defineInterfaceConfig("samplesPerSymbol",
                                    "Samples per Symbol",
                                    "Input samples per symbol. Changing it resets the eye history.",
                                    {{"type", "range"}, {"min", 1.0f}, {"max", 256.0f}, {"value_type", "uint"}}));

    JST_CHECK(defineInterfaceConfig("interpolation",
                                    "Interpolation",
                                    "Display points per input interval, using linear interpolation.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "1x (None)"}, {"value", "1"}},
                                        Parser::Map{{"label", "2x (Linear)"}, {"value", "2"}},
                                        Parser::Map{{"label", "4x (Linear)"}, {"value", "4"}},
                                        Parser::Map{{"label", "8x (Linear)"}, {"value", "8"}},
                                        Parser::Map{{"label", "16x (Linear)"}, {"value", "16"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("component",
                                    "Component",
                                    "Complex signal component.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "In-Phase (Real)"}, {"value", "real"}},
                                        Parser::Map{{"label", "Quadrature (Imaginary)"}, {"value", "imag"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("rangeMin",
                                    "Amplitude Min",
                                    "Lower amplitude limit.",
                                    {{"type", "float"}}));

    JST_CHECK(defineInterfaceConfig("rangeMax",
                                    "Amplitude Max",
                                    "Upper amplitude limit.",
                                    {{"type", "float"}}));

    JST_CHECK(defineInterfaceConfig("persistence",
                                    "Persistence",
                                    "Number of displayed traces for the density to fade to half.",
                                    {{"type", "range"}, {"min", 1.0f}, {"max", 4096.0f}, {"value_type", "uint"}, {"unit", "traces"}}));

    return Result::SUCCESS;
}

Result EyeDiagramImpl::create() {
    JST_CHECK(moduleCreate("eye_diagram", eyeDiagramConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleCreate("signal_view", signalViewConfig, {
        {"signal", moduleGetOutput({"eye_diagram", "signal"})}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(EyeDiagramImpl, {"eye_diagram"}, {"signal_view"});

}  // namespace Jetstream::Blocks
