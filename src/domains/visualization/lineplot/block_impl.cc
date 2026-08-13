#include <jetstream/domains/visualization/lineplot/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/signal_view/module.hh>

namespace Jetstream::Blocks {

struct LineplotImpl : public Block::Impl, public DynamicConfig<Blocks::Lineplot> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::SignalView> signalViewConfig =
        std::make_shared<Modules::SignalView>();
};

Result LineplotImpl::configure() {
    signalViewConfig->mode = "lineplot";
    signalViewConfig->averaging = averaging;
    signalViewConfig->decimation = decimation;
    signalViewConfig->maxHold = maxHold;
    signalViewConfig->fill = fill;
    signalViewConfig->rangeMin = rangeMin;
    signalViewConfig->rangeMax = rangeMax;
    signalViewConfig->xLabel = xLabel;
    signalViewConfig->amplitudeLabel = yLabel;

    return Result::SUCCESS;
}

Result LineplotImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Input signal data to visualize."));

    JST_CHECK(defineInterfaceConfig("averaging",
                                    "Averaging",
                                    "Number of samples to average for smoothing.",
                                    "range:1:256:samples:uint"));

    JST_CHECK(defineInterfaceConfig("decimation",
                                    "Decimation",
                                    "Decimation factor for input data.",
                                    "range:1:64::uint"));

    JST_CHECK(defineInterfaceConfig("maxHold",
                                    "Max Hold",
                                    "Enable maximum hold trace.",
                                    "bool"));

    return Result::SUCCESS;
}

Result LineplotImpl::create() {
    JST_CHECK(moduleCreate("signal_view", signalViewConfig, {
        {"signal", inputs().at("signal")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(LineplotImpl, {"signal_view"});

}  // namespace Jetstream::Blocks
