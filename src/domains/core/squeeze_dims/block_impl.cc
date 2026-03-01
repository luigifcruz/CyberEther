#include <jetstream/domains/core/squeeze_dims/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/squeeze_dims/module.hh>

namespace Jetstream::Blocks {

struct SqueezeDimsImpl : public Block::Impl, public DynamicConfig<Blocks::SqueezeDims> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::SqueezeDims> squeezeDimsModuleConfig =
        std::make_shared<Modules::SqueezeDims>();
};

Result SqueezeDimsImpl::configure() {
    squeezeDimsModuleConfig->axis = axis;

    return Result::SUCCESS;
}

Result SqueezeDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to squeeze."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Tensor with squeezed dimension."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Position of the dimension to remove (must have size 1).",
                                    "int:"));

    return Result::SUCCESS;
}

Result SqueezeDimsImpl::create() {
    JST_CHECK(moduleCreate("squeeze_dims", squeezeDimsModuleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    JST_CHECK(moduleExposeOutput("buffer", {"squeeze_dims", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SqueezeDimsImpl);

}  // namespace Jetstream::Blocks
