#include <jetstream/domains/core/permutation/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/permutation/module.hh>

namespace Jetstream::Blocks {

struct PermutationImpl : public Block::Impl, public DynamicConfig<Blocks::Permutation> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Permutation> moduleConfig = std::make_shared<Modules::Permutation>();
};

Result PermutationImpl::configure() {
    moduleConfig->permutation = permutation;

    return Result::SUCCESS;
}

Result PermutationImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to reorder."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Tensor view with permuted axes."));

    JST_CHECK(defineInterfaceConfig("permutation",
                                    "Permutation",
                                    "Output axis order as zero-based input axis indices.",
                                    "vector-inline:int:axis"));

    return Result::SUCCESS;
}

Result PermutationImpl::create() {
    JST_CHECK(moduleCreate("permutation", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"permutation", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PermutationImpl);

}  // namespace Jetstream::Blocks
