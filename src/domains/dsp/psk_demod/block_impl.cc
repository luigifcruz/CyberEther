#include <jetstream/domains/dsp/psk_demod/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/psk_demod/module.hh>

namespace Jetstream::Blocks {

struct PskDemodImpl : public Block::Impl, public DynamicConfig<Blocks::PskDemod> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::PskDemod> pskDemodConfig = std::make_shared<Modules::PskDemod>();
};

Result PskDemodImpl::configure() {
    pskDemodConfig->pskType = pskType;
    pskDemodConfig->sampleRate = sampleRate;
    pskDemodConfig->symbolRate = symbolRate;
    pskDemodConfig->frequencyLoopBandwidth = frequencyLoopBandwidth;
    pskDemodConfig->timingLoopBandwidth = timingLoopBandwidth;
    pskDemodConfig->dampingFactor = dampingFactor;

    return Result::SUCCESS;
}

Result PskDemodImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Complex input signal (IQ samples)."));

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Demodulated soft symbols."));

    JST_CHECK(defineInterfaceConfig("pskType",
                                    "PSK Type",
                                    "The PSK modulation scheme to demodulate.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "BPSK"}, {"value", "bpsk"}},
                                        Parser::Map{{"label", "QPSK"}, {"value", "qpsk"}},
                                        Parser::Map{{"label", "8-PSK"}, {"value", "8psk"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Input signal sample rate.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("symbolRate",
                                    "Symbol Rate",
                                    "Expected symbol rate.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("frequencyLoopBandwidth",
                                    "Freq Loop BW",
                                    "Carrier recovery loop bandwidth (0-1).",
                                    {{"type", "range"}, {"min", 0.001f}, {"max", 0.2f}}));

    JST_CHECK(defineInterfaceConfig("timingLoopBandwidth",
                                    "Timing Loop BW",
                                    "Symbol timing recovery loop bandwidth (0-1).",
                                    {{"type", "range"}, {"min", 0.001f}, {"max", 0.2f}}));

    JST_CHECK(defineInterfaceConfig("dampingFactor",
                                    "Damping Factor",
                                    "Loop filter damping coefficient.",
                                    {{"type", "range"}, {"min", 0.1f}, {"max", 2.0f}}));

    return Result::SUCCESS;
}

Result PskDemodImpl::create() {
    JST_CHECK(moduleCreate("psk_demod", pskDemodConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"psk_demod", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PskDemodImpl, {"psk_demod"});

}  // namespace Jetstream::Blocks
