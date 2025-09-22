#include <jetstream/domains/visualization/waterfall/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/waterfall/module.hh>

namespace Jetstream::Blocks {

struct WaterfallImpl : public Block::Impl, public DynamicConfig<Blocks::Waterfall> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Waterfall> waterfallConfig = std::make_shared<Modules::Waterfall>();
};

Result WaterfallImpl::configure() {
    waterfallConfig->height = height;
    waterfallConfig->interpolate = interpolate;

    return Result::SUCCESS;
}

Result WaterfallImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Input signal data to visualize."));

    JST_CHECK(defineInterfaceConfig("height",
                                    "Height",
                                    "Number of rows in the waterfall history buffer.",
                                    "int:rows"));

    JST_CHECK(defineInterfaceConfig("interpolate",
                                    "Interpolate",
                                    "Enable texture interpolation for smoother display.",
                                    "bool"));

    return Result::SUCCESS;
}

Result WaterfallImpl::create() {
    JST_CHECK(moduleCreate("waterfall", waterfallConfig, {
        {"signal", inputs().at("signal")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(WaterfallImpl);

}  // namespace Jetstream::Blocks
