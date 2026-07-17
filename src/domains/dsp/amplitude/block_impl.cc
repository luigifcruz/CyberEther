#include <jetstream/domains/dsp/amplitude/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/amplitude/module.hh>

namespace Jetstream::Blocks {

struct AmplitudeImpl : public Block::Impl, public DynamicConfig<Blocks::Amplitude> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Amplitude> amplitudeConfig = std::make_shared<Modules::Amplitude>();
};

Result AmplitudeImpl::configure() {
    amplitudeConfig->axis = axis;
    return Result::SUCCESS;
}

Result AmplitudeImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Input signal (complex or real)."));

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Amplitude output in decibels."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Axis whose length is used for normalization. "
                                    "Negative axes count from the end.",
                                    "int:"));

    return Result::SUCCESS;
}

Result AmplitudeImpl::create() {
    JST_CHECK(moduleCreate("amplitude", amplitudeConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"amplitude", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AmplitudeImpl);

}  // namespace Jetstream::Blocks
