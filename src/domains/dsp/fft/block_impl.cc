#include <jetstream/domains/dsp/fft/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/fft/module.hh>

namespace Jetstream::Blocks {

struct FftImpl : public Block::Impl, public DynamicConfig<Blocks::Fft> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Fft> fftConfig = std::make_shared<Modules::Fft>();
};

Result FftImpl::configure() {
    fftConfig->forward = forward;

    return Result::SUCCESS;
}

Result FftImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Input signal to transform."));

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Transformed signal output."));

    JST_CHECK(defineInterfaceConfig("forward",
                                    "Direction",
                                    "Transform direction: Forward converts time-domain to "
                                    "frequency-domain, Inverse converts back.",
                                    "dropdown:true(Forward),false(Inverse)"));

    return Result::SUCCESS;
}

Result FftImpl::create() {
    JST_CHECK(moduleCreate("fft", fftConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"fft", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FftImpl);

}  // namespace Jetstream::Blocks
