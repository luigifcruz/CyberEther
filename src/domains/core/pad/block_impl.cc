#include <jetstream/domains/core/pad/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/pad/module.hh>

namespace Jetstream::Blocks {

struct PadImpl : public Block::Impl, public DynamicConfig<Blocks::Pad> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Pad> moduleConfig = std::make_shared<Modules::Pad>();
};

Result PadImpl::configure() {
    moduleConfig->size = size;
    moduleConfig->axis = axis;

    return Result::SUCCESS;
}

Result PadImpl::define() {
    JST_CHECK(defineInterfaceInput("unpadded", "Input", "Input tensor to pad."));
    JST_CHECK(defineInterfaceOutput("padded", "Output", "Padded output tensor."));

    JST_CHECK(defineInterfaceConfig("size",
                                    "Pad Size",
                                    "Number of zeros to append.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Pad Axis",
                                    "Dimension along which to add padding.",
                                    "int:"));

    return Result::SUCCESS;
}

Result PadImpl::create() {
    JST_CHECK(moduleCreate("pad", moduleConfig, {
        {"unpadded", inputs().at("unpadded")}
    }));
    JST_CHECK(moduleExposeOutput("padded", {"pad", "padded"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PadImpl);

}  // namespace Jetstream::Blocks
