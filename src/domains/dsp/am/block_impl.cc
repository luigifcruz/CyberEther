#include <jetstream/domains/dsp/am/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/am/module.hh>

namespace Jetstream::Blocks {

struct AmImpl : public Block::Impl, public DynamicConfig<Blocks::AM> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::AM> amConfig = std::make_shared<Modules::AM>();
};

Result AmImpl::configure() {
    amConfig->sampleRate = sampleRate;
    amConfig->dcAlpha = dcAlpha;

    return Result::SUCCESS;
}

Result AmImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Complex input signal (IQ samples)."));

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Demodulated audio signal."));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Input signal sample rate.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("dcAlpha",
                                    "DC Alpha",
                                    "DC-blocking filter coefficient.",
                                    {{"type", "range"}, {"min", 0.9f}, {"max", 0.999f}, {"unit", "ratio"}}));

    return Result::SUCCESS;
}

Result AmImpl::create() {
    JST_CHECK(moduleCreate("am", amConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"am", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AmImpl, {"am"});

}  // namespace Jetstream::Blocks
