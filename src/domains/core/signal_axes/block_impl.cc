#include <jetstream/domains/core/signal_axes/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/signal_axes/module.hh>

namespace Jetstream::Blocks {

struct SignalAxesImpl : public Block::Impl,
                        public DynamicConfig<Blocks::SignalAxes> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::SignalAxes> moduleConfig =
        std::make_shared<Modules::SignalAxes>();
};

Result SignalAxesImpl::configure() {
    moduleConfig->axes = axes;
    return Result::SUCCESS;
}

Result SignalAxesImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Tensor to relabel."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Tensor with updated signal axes."));

    JST_CHECK(defineInterfaceConfig(
        "axes",
        "Axes",
        "Axis roles in [B, C, S, _, *] notation. * preserves an input role.",
        "text"));

    return Result::SUCCESS;
}

Result SignalAxesImpl::create() {
    JST_CHECK(moduleCreate("signal_axes", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"signal_axes", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SignalAxesImpl, {"signal_axes"});

}  // namespace Jetstream::Blocks
