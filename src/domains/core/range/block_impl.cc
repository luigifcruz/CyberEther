#include <jetstream/domains/core/range/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/range/module.hh>

namespace Jetstream::Blocks {

struct RangeImpl : public Block::Impl, public DynamicConfig<Blocks::Range> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Range> moduleConfig = std::make_shared<Modules::Range>();
};

Result RangeImpl::configure() {
    moduleConfig->min = min;
    moduleConfig->max = max;

    return Result::SUCCESS;
}

Result RangeImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Scaled output signal."));

    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Input signal to be scaled."));

    JST_CHECK(defineInterfaceConfig("min",
                                    "Min",
                                    "Minimum value of the input range.",
                                    "range:-100:0:dBFS:float"));

    JST_CHECK(defineInterfaceConfig("max",
                                    "Max",
                                    "Maximum value of the input range.",
                                    "range:-100:0:dBFS:float"));

    return Result::SUCCESS;
}

Result RangeImpl::create() {
    JST_CHECK(moduleCreate("range", moduleConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"range", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(RangeImpl);

}  // namespace Jetstream::Blocks
