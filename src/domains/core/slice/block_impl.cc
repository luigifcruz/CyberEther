#include <jetstream/domains/core/slice/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/slice/module.hh>
#include <jetstream/domains/core/duplicate/module.hh>

namespace Jetstream::Blocks {

struct SliceImpl : public Block::Impl, public DynamicConfig<Blocks::Slice> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Slice> sliceModuleConfig = std::make_shared<Modules::Slice>();
    std::shared_ptr<Modules::Duplicate> duplicateModuleConfig = std::make_shared<Modules::Duplicate>();
};

Result SliceImpl::validate() {
    const auto& config = *candidate();

    if (contiguous != config.contiguous) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result SliceImpl::configure() {
    sliceModuleConfig->slice = slice;
    duplicateModuleConfig->hostAccessible = true;

    return Result::SUCCESS;
}

Result SliceImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to slice."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Sliced output tensor."));

    JST_CHECK(defineInterfaceConfig("slice",
                                    "Slice",
                                    "NumPy-style slice notation (e.g., [0:10, ...]).",
                                    "text"));

    JST_CHECK(defineInterfaceConfig("contiguous",
                                    "Contiguous",
                                    "Copy data to ensure contiguous memory layout.",
                                    "bool"));

    return Result::SUCCESS;
}

Result SliceImpl::create() {
    JST_CHECK(moduleCreate("slice", sliceModuleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    if (contiguous) {
        JST_CHECK(moduleCreate("duplicate", duplicateModuleConfig, {
            {"buffer", moduleGetOutput({"slice", "buffer"})}
        }));
        JST_CHECK(moduleExposeOutput("buffer", {"duplicate", "buffer"}));
    } else {
        JST_CHECK(moduleExposeOutput("buffer", {"slice", "buffer"}));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SliceImpl);

}  // namespace Jetstream::Blocks
