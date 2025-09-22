#include <jetstream/domains/core/unpad/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/unpad/module.hh>

namespace Jetstream::Blocks {

struct UnpadImpl : public Block::Impl, public DynamicConfig<Blocks::Unpad> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Unpad> moduleConfig = std::make_shared<Modules::Unpad>();
};

Result UnpadImpl::configure() {
    moduleConfig->size = size;
    moduleConfig->axis = axis;

    return Result::SUCCESS;
}

Result UnpadImpl::define() {
    JST_CHECK(defineInterfaceInput("padded", "Input", "Input tensor with padding."));
    JST_CHECK(defineInterfaceOutput("unpadded", "Output", "Tensor with padding removed."));
    JST_CHECK(defineInterfaceOutput("pad", "Pad", "The removed padding portion."));

    JST_CHECK(defineInterfaceConfig("size",
                                    "Pad Size",
                                    "Number of elements to remove.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Pad Axis",
                                    "Dimension along which to remove padding.",
                                    "int:"));

    return Result::SUCCESS;
}

Result UnpadImpl::create() {
    JST_CHECK(moduleCreate("unpad", moduleConfig, {
        {"padded", inputs().at("padded")}
    }));
    JST_CHECK(moduleExposeOutput("unpadded", {"unpad", "unpadded"}));
    JST_CHECK(moduleExposeOutput("pad", {"unpad", "pad"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(UnpadImpl);

}  // namespace Jetstream::Blocks
