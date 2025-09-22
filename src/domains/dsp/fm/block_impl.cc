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

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Input signal sample rate.",
                                    "float:MHz:3"));

    return Result::SUCCESS;
}

Result FmImpl::create() {
    JST_CHECK(moduleCreate("fm", fmConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"fm", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FmImpl);

}  // namespace Jetstream::Blocks
