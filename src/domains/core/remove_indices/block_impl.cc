#include <jetstream/domains/core/remove_indices/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/remove_indices/module.hh>

namespace Jetstream::Blocks {

struct RemoveIndicesImpl : public Block::Impl, public DynamicConfig<Blocks::RemoveIndices> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::RemoveIndices> moduleConfig = std::make_shared<Modules::RemoveIndices>();
};

Result RemoveIndicesImpl::configure() {
    moduleConfig->axis = axis;
    moduleConfig->indices = indices;
    return Result::SUCCESS;
}

Result RemoveIndicesImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to remove entries from."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Contiguous tensor with selected entries removed."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Dimension to remove entries from. Negative axes count from the end.",
                                    {{"type", "int"}}));

    JST_CHECK(defineInterfaceConfig("indices",
                                    "Indices",
                                    "Zero-based indices to remove from the original input. Duplicates count once.",
                                    {{"type", "vector-inline"}, {"value_type", "uint"}}));

    return Result::SUCCESS;
}

Result RemoveIndicesImpl::create() {
    JST_CHECK(moduleCreate("remove_indices", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"remove_indices", "buffer"}));
    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(RemoveIndicesImpl, {"remove_indices"});

}  // namespace Jetstream::Blocks
