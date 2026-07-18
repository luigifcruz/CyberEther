#include <jetstream/domains/core/flatten/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/duplicate/module.hh>
#include <jetstream/domains/core/flatten/module.hh>

namespace Jetstream::Blocks {

struct FlattenImpl : public Block::Impl, public DynamicConfig<Blocks::Flatten> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Flatten> flattenModuleConfig = std::make_shared<Modules::Flatten>();
    std::shared_ptr<Modules::Duplicate> duplicateModuleConfig = std::make_shared<Modules::Duplicate>();
};

Result FlattenImpl::validate() {
    const auto& config = *candidate();

    if (contiguous != config.contiguous) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result FlattenImpl::configure() {
    duplicateModuleConfig->hostAccessible = true;

    return Result::SUCCESS;
}

Result FlattenImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to flatten."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Flattened 1D output tensor."));

    JST_CHECK(defineInterfaceConfig("contiguous",
                                    "Contiguous",
                                    "Copy data to ensure contiguous memory layout before flattening.",
                                    "bool"));

    return Result::SUCCESS;
}

Result FlattenImpl::create() {
    if (contiguous) {
        JST_CHECK(moduleCreate("duplicate", duplicateModuleConfig, {
            {"buffer", inputs().at("buffer")}
        }));
        JST_CHECK(moduleCreate("flatten", flattenModuleConfig, {
            {"buffer", moduleGetOutput({"duplicate", "buffer"})}
        }));
    } else {
        JST_CHECK(moduleCreate("flatten", flattenModuleConfig, {
            {"buffer", inputs().at("buffer")}
        }));
    }

    JST_CHECK(moduleExposeOutput("buffer", {"flatten", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FlattenImpl,
                   {"flatten"},
                   {"duplicate", true});

}  // namespace Jetstream::Blocks
