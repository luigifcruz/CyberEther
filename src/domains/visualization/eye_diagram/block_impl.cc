#include <jetstream/domains/visualization/eye_diagram/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <jetstream/domains/visualization/eye_diagram/module.hh>
#include <jetstream/domains/visualization/signal_view/module.hh>

namespace Jetstream::Blocks {

struct EyeDiagramImpl : public Block::Impl, public DynamicConfig<Blocks::EyeDiagram> {
    Result configure() override;
    Result define() override;
    Result create() override;

    std::shared_ptr<Modules::EyeDiagram> eyeConfig =
        std::make_shared<Modules::EyeDiagram>();
    std::shared_ptr<Modules::SignalView> plotConfig =
        std::make_shared<Modules::SignalView>();
};

Result EyeDiagramImpl::configure() {
    eyeConfig->samplesPerSymbol = samplesPerSymbol;
    eyeConfig->interpolation = interpolation;
    eyeConfig->component = component;
    eyeConfig->rangeMin = rangeMin;
    eyeConfig->rangeMax = rangeMax;
    plotConfig->mode = "lineplot";
    plotConfig->fill = false;
    plotConfig->density = true;
    plotConfig->persistence = persistence;
    plotConfig->rangeMin = rangeMin;
    plotConfig->rangeMax = rangeMax;
    plotConfig->xLabel = "Time (symbols)";
    plotConfig->amplitudeLabel = "Amplitude";
    return Result::SUCCESS;
}

Result EyeDiagramImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input",
                                  "Oversampled F32 or CF32 signal."));
    JST_CHECK(defineInterfaceConfig("samplesPerSymbol", "Samples per Symbol",
        "Input samples per symbol; resets the eye history when changed.",
        {{"type", "range"}, {"min", 1.0f}, {"max", 256.0f},
         {"value_type", "uint"}}));
    JST_CHECK(defineInterfaceConfig("interpolation", "Interpolation",
        "Display points per input interval (linear interpolation).",
        {{"type", "dropdown"}, {"options", Parser::Sequence{
            Parser::Map{{"label", "1x (None)"}, {"value", "1"}},
            Parser::Map{{"label", "2x (Linear)"}, {"value", "2"}},
            Parser::Map{{"label", "4x (Linear)"}, {"value", "4"}},
            Parser::Map{{"label", "8x (Linear)"}, {"value", "8"}},
            Parser::Map{{"label", "16x (Linear)"}, {"value", "16"}},
        }}}));
    JST_CHECK(defineInterfaceConfig("component", "Component",
        "Complex signal component.",
        {{"type", "dropdown"}, {"options", Parser::Sequence{
            Parser::Map{{"label", "In-Phase (Real)"}, {"value", "real"}},
            Parser::Map{{"label", "Quadrature (Imaginary)"}, {"value", "imag"}},
        }}}));
    JST_CHECK(defineInterfaceConfig("rangeMin", "Amplitude Min",
        "Lower amplitude limit.", {{"type", "float"}}));
    JST_CHECK(defineInterfaceConfig("rangeMax", "Amplitude Max",
        "Upper amplitude limit.", {{"type", "float"}}));
    JST_CHECK(defineInterfaceConfig("persistence", "Persistence",
        "Number of displayed traces for the density to fade to half.",
        {{"type", "range"}, {"min", 1.0f}, {"max", 4096.0f},
         {"value_type", "uint"}, {"unit", "traces"}}));
    return Result::SUCCESS;
}

Result EyeDiagramImpl::create() {
    JST_CHECK(moduleCreate("eye_diagram", eyeConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleCreate("signal_view", plotConfig, {
        {"signal", moduleGetOutput({"eye_diagram", "signal"})}
    }));
    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(EyeDiagramImpl, {"eye_diagram"}, {"signal_view"});

}  // namespace Jetstream::Blocks
