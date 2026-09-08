#include <jetstream/domains/dsp/fm/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/fm/module.hh>

namespace Jetstream::Blocks {

struct FmImpl : public Block::Impl, public DynamicConfig<Blocks::FM> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::FM> fmConfig = std::make_shared<Modules::FM>();
};

Result FmImpl::configure() {
    fmConfig->mode = mode;
    fmConfig->deemphasis = deemphasis;
    fmConfig->sampleRate = sampleRate;

    return Result::SUCCESS;
}

Result FmImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Complex input signal (IQ samples)."));

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Demodulated audio signal."));

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Select mono narrowband or stereo wideband FM.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Narrowband"}, {"value", "narrow"}},
                                        Parser::Map{{"label", "Wideband"}, {"value", "wide"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("deemphasis",
                                    "De-emphasis",
                                    "Optional FM de-emphasis time constant.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "None"}, {"value", "none"}},
                                        Parser::Map{{"label", "50 us [Global]"}, {"value", "50us"}},
                                        Parser::Map{{"label", "75 us [USA]"}, {"value", "75us"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Input signal sample rate.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    return Result::SUCCESS;
}

Result FmImpl::create() {
    JST_CHECK(moduleCreate("fm", fmConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"fm", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FmImpl, {"fm"});

}  // namespace Jetstream::Blocks
