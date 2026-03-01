#include <jetstream/domains/core/reshape/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/core/duplicate/module.hh>

namespace Jetstream::Blocks {

struct ReshapeImpl : public Block::Impl, public DynamicConfig<Blocks::Reshape> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Reshape> reshapeModuleConfig = std::make_shared<Modules::Reshape>();
    std::shared_ptr<Modules::Duplicate> duplicateModuleConfig = std::make_shared<Modules::Duplicate>();
};

Result ReshapeImpl::validate() {
    const auto& config = *candidate();

    if (contiguous != config.contiguous) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result ReshapeImpl::configure() {
    reshapeModuleConfig->shape = shape;
    duplicateModuleConfig->hostAccessible = true;

    return Result::SUCCESS;
}

Result ReshapeImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to reshape."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Reshaped output tensor."));

    JST_CHECK(defineInterfaceConfig("shape",
                                    "Shape",
                                    "Target shape as bracket notation (e.g., [100, 200]).",
                                    "text"));

    JST_CHECK(defineInterfaceConfig("contiguous",
                                    "Contiguous",
                                    "Copy data to ensure contiguous memory layout before reshaping.",
                                    "bool"));

    return Result::SUCCESS;
}

Result ReshapeImpl::create() {
    if (contiguous) {
        JST_CHECK(moduleCreate("duplicate", duplicateModuleConfig, {
            {"buffer", inputs().at("buffer")}
        }));
        JST_CHECK(moduleCreate("reshape", reshapeModuleConfig, {
            {"buffer", moduleGetOutput({"duplicate", "buffer"})}
        }));
    } else {
        JST_CHECK(moduleCreate("reshape", reshapeModuleConfig, {
            {"buffer", inputs().at("buffer")}
        }));
    }

    JST_CHECK(moduleExposeOutput("buffer", {"reshape", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ReshapeImpl);

}  // namespace Jetstream::Blocks
