#include <jetstream/domains/visualization/constellation/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/constellation/module.hh>

namespace Jetstream::Blocks {

struct ConstellationImpl : public Block::Impl,
                           public DynamicConfig<Blocks::Constellation> {
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Constellation> constellationConfig =
        std::make_shared<Modules::Constellation>();
};

Result ConstellationImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Complex-valued input signal."));

    return Result::SUCCESS;
}

Result ConstellationImpl::create() {
    JST_CHECK(moduleCreate("constellation", constellationConfig, {
        {"signal", inputs().at("signal")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ConstellationImpl);

}  // namespace Jetstream::Blocks
