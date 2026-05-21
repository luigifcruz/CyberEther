#include <blueprint/gain/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blueprint/gain/module.hh>

namespace Jetstream::Blocks {

struct BlueprintGainImpl : public Block::Impl, public DynamicConfig<Blocks::BlueprintGain> {
    Result configure() override;
    Result define() override;
    Result create() override;

  protected:
    std::shared_ptr<Modules::BlueprintGain> moduleConfig = std::make_shared<Modules::BlueprintGain>();
};

Result BlueprintGainImpl::configure() {
    moduleConfig->gain = gain;

    return Result::SUCCESS;
}

Result BlueprintGainImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Signal", "Input samples to scale."));
    JST_CHECK(defineInterfaceOutput("signal", "Signal", "Scaled output samples."));

    JST_CHECK(defineInterfaceConfig("gain",
                                    "Gain",
                                    "Scalar multiplier applied to each sample.",
                                    "float::3"));

    return Result::SUCCESS;
}

Result BlueprintGainImpl::create() {
    JST_CHECK(moduleCreate("blueprint_gain", moduleConfig, {
        {"signal", inputs().at("signal")},
    }));
    JST_CHECK(moduleExposeOutput("signal", {"blueprint_gain", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(BlueprintGainImpl);

}  // namespace Jetstream::Blocks
