#include <jetstream/domains/core/ones_tensor/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/ones_tensor/module.hh>

namespace Jetstream::Blocks {

struct OnesTensorImpl : public Block::Impl, public DynamicConfig<Blocks::OnesTensor> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::OnesTensor> moduleConfig = std::make_shared<Modules::OnesTensor>();
};

Result OnesTensorImpl::configure() {
    moduleConfig->shape = shape;
    moduleConfig->dataType = dataType;

    return Result::SUCCESS;
}

Result OnesTensorImpl::define() {
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Tensor of ones."));

    JST_CHECK(defineInterfaceConfig("shape",
                                    "Shape",
                                    "Output tensor shape as a list of positive dimensions.",
                                    {{"type", "vector-inline"}, {"value_type", "uint"}, {"unit", "dim"}}));

    JST_CHECK(defineInterfaceConfig("dataType",
                                    "Data Type",
                                    "Output tensor type.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "F32"}, {"value", "F32"}},
                                        Parser::Map{{"label", "CF32"}, {"value", "CF32"}},
                                        Parser::Map{{"label", "F64"}, {"value", "F64"}},
                                        Parser::Map{{"label", "CF64"}, {"value", "CF64"}},
                                    }}}));

    return Result::SUCCESS;
}

Result OnesTensorImpl::create() {
    JST_CHECK(moduleCreate("ones_tensor", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("buffer", {"ones_tensor", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(OnesTensorImpl, {"ones_tensor"});

}  // namespace Jetstream::Blocks
