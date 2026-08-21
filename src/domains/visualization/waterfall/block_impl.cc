#include <jetstream/domains/visualization/waterfall/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/signal_view/module.hh>

namespace Jetstream::Blocks {

struct WaterfallImpl : public Block::Impl, public DynamicConfig<Blocks::Waterfall> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::SignalView> signalViewConfig =
        std::make_shared<Modules::SignalView>();
};

Result WaterfallImpl::configure() {
    signalViewConfig->mode = "waterfall";
    signalViewConfig->waterfallHeight = height;
    signalViewConfig->xLabel = xLabel;
    signalViewConfig->waterfallLabel = yLabel;

    return Result::SUCCESS;
}

Result WaterfallImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Input signal data to visualize."));

    JST_CHECK(defineInterfaceConfig("height",
                                    "Height",
                                    "Number of rows in the waterfall history buffer.",
                                    "uint:rows"));

    return Result::SUCCESS;
}

Result WaterfallImpl::create() {
    JST_CHECK(moduleCreate("signal_view", signalViewConfig, {
        {"signal", inputs().at("signal")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(WaterfallImpl, {"signal_view"});

}  // namespace Jetstream::Blocks
