#include <jetstream/domains/dsp/agc/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/dsp/agc/module.hh>

namespace Jetstream::Blocks {

struct AgcImpl : public Block::Impl, public DynamicConfig<Blocks::Agc> {
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Agc> moduleConfig = std::make_shared<Modules::Agc>();
};

Result AgcImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Signal to be normalized."));
    JST_CHECK(defineInterfaceOutput("signal", "Output", "Normalized signal."));

    return Result::SUCCESS;
}

Result AgcImpl::create() {
    JST_CHECK(moduleCreate("agc", moduleConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"agc", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AgcImpl);

}  // namespace Jetstream::Blocks
