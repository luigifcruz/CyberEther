#include <jetstream/domains/core/expand_dims/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/expand_dims/module.hh>

namespace Jetstream::Blocks {

struct ExpandDimsImpl : public Block::Impl, public DynamicConfig<Blocks::ExpandDims> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::ExpandDims> expandDimsModuleConfig =
        std::make_shared<Modules::ExpandDims>();
};

Result ExpandDimsImpl::configure() {
    expandDimsModuleConfig->axis = axis;

    return Result::SUCCESS;
}

Result ExpandDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to expand."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Tensor with expanded dimension."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Position to insert the new dimension (0-indexed).",
                                    "int:"));

    return Result::SUCCESS;
}

Result ExpandDimsImpl::create() {
    JST_CHECK(moduleCreate("expand_dims", expandDimsModuleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    JST_CHECK(moduleExposeOutput("buffer", {"expand_dims", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ExpandDimsImpl);

}  // namespace Jetstream::Blocks
