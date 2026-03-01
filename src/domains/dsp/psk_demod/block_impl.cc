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
                                    "dropdown:bpsk(BPSK),qpsk(QPSK),8psk(8-PSK)"));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Input signal sample rate.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("symbolRate",
                                    "Symbol Rate",
                                    "Expected symbol rate.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("frequencyLoopBandwidth",
                                    "Freq Loop BW",
                                    "Carrier recovery loop bandwidth (0-1).",
                                    "range:0.001:0.2::float"));

    JST_CHECK(defineInterfaceConfig("timingLoopBandwidth",
                                    "Timing Loop BW",
                                    "Symbol timing recovery loop bandwidth (0-1).",
                                    "range:0.001:0.2::float"));

    JST_CHECK(defineInterfaceConfig("dampingFactor",
                                    "Damping Factor",
                                    "Loop filter damping coefficient.",
                                    "range:0.1:2.0::float"));

    return Result::SUCCESS;
}

Result PskDemodImpl::create() {
    JST_CHECK(moduleCreate("psk_demod", pskDemodConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"psk_demod", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PskDemodImpl);

}  // namespace Jetstream::Blocks
