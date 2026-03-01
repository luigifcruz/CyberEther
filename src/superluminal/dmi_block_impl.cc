#include "dmi_block.hh"
#include <jetstream/detail/block_impl.hh>

#include "dmi_module.hh"

namespace Jetstream::Blocks {

struct DynamicTensorImportImpl : public Block::Impl,
                                 public DynamicConfig<Blocks::DynamicTensorImport> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::DynamicTensorImport> dtiConfig =
        std::make_shared<Modules::DynamicTensorImport>();
};

Result DynamicTensorImportImpl::configure() {
    dtiConfig->buffer = buffer;

    return Result::SUCCESS;
}

Result DynamicTensorImportImpl::define() {
    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Imported external tensor."));

    return Result::SUCCESS;
}

Result DynamicTensorImportImpl::create() {
    JST_CHECK(moduleCreate("source", dtiConfig, {}));
    JST_CHECK(moduleExposeOutput("buffer", {"source", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(DynamicTensorImportImpl);

}  // namespace Jetstream::Blocks
