#include <jetstream/domains/core/cast/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/cast/module.hh>

namespace Jetstream::Blocks {

struct CastImpl : public Block::Impl, public DynamicConfig<Blocks::Cast> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Cast> moduleConfig =
        std::make_shared<Modules::Cast>();
};

Result CastImpl::configure() {
    moduleConfig->outputType = outputType;

    return Result::SUCCESS;
}

Result CastImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input",
                                   "Input signal to cast."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output",
                                    "Cast output signal."));

    JST_CHECK(defineInterfaceConfig("outputType",
                                    "Output Type",
                                    "The desired output data type.",
                                    "dropdown:CF32(CF32),F32(F32)"));

    return Result::SUCCESS;
}

Result CastImpl::create() {
    JST_CHECK(moduleCreate("cast", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"cast", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(CastImpl);

}  // namespace Jetstream::Blocks
