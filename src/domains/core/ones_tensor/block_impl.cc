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
                                    "vector-inline:int:dim"));

    JST_CHECK(defineInterfaceConfig("dataType",
                                    "Data Type",
                                    "Output tensor type.",
                                    "dropdown:F32(F32),CF32(CF32)"));

    return Result::SUCCESS;
}

Result OnesTensorImpl::create() {
    JST_CHECK(moduleCreate("ones_tensor", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("buffer", {"ones_tensor", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(OnesTensorImpl);

}  // namespace Jetstream::Blocks
