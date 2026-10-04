#include <jetstream/domains/visualization/waterfall_3d/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/signal_view/module.hh>
#include <jetstream/render/colormap.hh>

namespace Jetstream::Blocks {

struct Waterfall3DImpl : public Block::Impl, public DynamicConfig<Blocks::Waterfall3D> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::SignalView> signalViewConfig =
        std::make_shared<Modules::SignalView>();
};

Result Waterfall3DImpl::configure() {
    signalViewConfig->mode = "waterfall_3d";
    signalViewConfig->waterfallAveraging = averaging;
    signalViewConfig->waterfallHeight = height;
    signalViewConfig->colormap = colormap;
    signalViewConfig->rangeMin = 0.0f;
    signalViewConfig->rangeMax = 1.0f;
    signalViewConfig->xLabel = xLabel;
    signalViewConfig->waterfallLabel = timeLabel;
    signalViewConfig->amplitudeLabel = amplitudeLabel;

    return Result::SUCCESS;
}

Result Waterfall3DImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Input signal data to visualize."));

    JST_CHECK(defineInterfaceConfig("height",
                                    "Height",
                                    "Number of rows in the history buffer.",
                                    {{"type", "uint"}, {"unit", "rows"}}));

    JST_CHECK(defineInterfaceConfig("averaging",
                                    "Averaging",
                                    "Number of spectra averaged per displayed row.",
                                    {{"type", "range"}, {"min", 1.0f}, {"max", 64.0f}, {"value_type", "uint"}}));

    JST_CHECK(defineInterfaceConfig("colormap",
                                    "Colormap",
                                    "Color palette for amplitude.",
                                    Render::Colormap::Format()));

    return Result::SUCCESS;
}

Result Waterfall3DImpl::create() {
    JST_CHECK(moduleCreate("signal_view", signalViewConfig, {
        {"signal", inputs().at("signal")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(Waterfall3DImpl, {"signal_view"});

}  // namespace Jetstream::Blocks
