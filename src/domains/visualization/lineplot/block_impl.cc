#include <jetstream/domains/visualization/lineplot/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/lineplot/module.hh>

namespace Jetstream::Blocks {

struct LineplotImpl : public Block::Impl, public DynamicConfig<Blocks::Lineplot> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Lineplot> lineplotConfig = std::make_shared<Modules::Lineplot>();
};

Result LineplotImpl::configure() {
    lineplotConfig->averaging = averaging;
    lineplotConfig->decimation = decimation;
    lineplotConfig->numberOfVerticalLines = numberOfVerticalLines;
    lineplotConfig->numberOfHorizontalLines = numberOfHorizontalLines;
    lineplotConfig->thickness = thickness;

    return Result::SUCCESS;
}

Result LineplotImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Input signal data to visualize."));

    JST_CHECK(defineInterfaceConfig("averaging",
                                    "Averaging",
                                    "Number of samples to average for smoothing.",
                                    "range:1:256:samples:int"));

    JST_CHECK(defineInterfaceConfig("decimation",
                                    "Decimation",
                                    "Decimation factor for input data.",
                                    "range:1:64::int"));

    return Result::SUCCESS;
}

Result LineplotImpl::create() {
    JST_CHECK(moduleCreate("lineplot", lineplotConfig, {
        {"signal", inputs().at("signal")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(LineplotImpl);

}  // namespace Jetstream::Blocks
