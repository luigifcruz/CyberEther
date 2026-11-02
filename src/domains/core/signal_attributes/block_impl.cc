#include <jetstream/domains/core/signal_attributes/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/signal_attributes/module.hh>

namespace Jetstream::Blocks {

struct SignalAttributesImpl : public Block::Impl,
                              public DynamicConfig<Blocks::SignalAttributes> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::SignalAttributes> moduleConfig =
        std::make_shared<Modules::SignalAttributes>();
};

Result SignalAttributesImpl::configure() {
    moduleConfig->sampleRate = sampleRate;
    moduleConfig->frequency = frequency;
    return Result::SUCCESS;
}

Result SignalAttributesImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Signal to tag."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Signal with the updated attributes."));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Sample rate to set. Zero keeps the input value.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("frequency",
                                    "Frequency",
                                    "Center frequency to set. Zero keeps the input value.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    return Result::SUCCESS;
}

Result SignalAttributesImpl::create() {
    JST_CHECK(moduleCreate("signal_attributes", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"signal_attributes", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SignalAttributesImpl, {"signal_attributes"});

}  // namespace Jetstream::Blocks
