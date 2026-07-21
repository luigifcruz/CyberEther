#include <jetstream/domains/core/invert/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/invert/module.hh>

namespace Jetstream::Blocks {

struct InvertImpl : public Block::Impl, public DynamicConfig<Blocks::Invert> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Invert> moduleConfig = std::make_shared<Modules::Invert>();
};

Result InvertImpl::configure() {
    moduleConfig->axis = axis;
    return Result::SUCCESS;
}

Result InvertImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Signal to be inverted."));
    JST_CHECK(defineInterfaceOutput("signal", "Output", "Inverted signal."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Axis along which to alternate signs. Negative axes count from the end.",
                                    "int:"));

    return Result::SUCCESS;
}

Result InvertImpl::create() {
    JST_CHECK(moduleCreate("invert", moduleConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"invert", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(InvertImpl, {"invert"});

}  // namespace Jetstream::Blocks
