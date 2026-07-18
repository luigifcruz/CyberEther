#include <jetstream/domains/core/permutation/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/duplicate/module.hh>
#include <jetstream/domains/core/permutation/module.hh>

namespace Jetstream::Blocks {

struct PermutationImpl : public Block::Impl, public DynamicConfig<Blocks::Permutation> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Permutation> permutationModuleConfig = std::make_shared<Modules::Permutation>();
    std::shared_ptr<Modules::Duplicate> duplicateModuleConfig = std::make_shared<Modules::Duplicate>();
};

Result PermutationImpl::validate() {
    const auto& config = *candidate();

    if (contiguous != config.contiguous) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result PermutationImpl::configure() {
    permutationModuleConfig->permutation = permutation;
    duplicateModuleConfig->hostAccessible = true;

    return Result::SUCCESS;
}

Result PermutationImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input tensor to reorder."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Tensor view with permuted axes."));

    JST_CHECK(defineInterfaceConfig("permutation",
                                    "Permutation",
                                    "Output axis order as zero-based input axis indices.",
                                    "vector-inline:uint:axis"));

    JST_CHECK(defineInterfaceConfig("contiguous",
                                    "Contiguous",
                                    "Copy data to ensure contiguous memory layout.",
                                    "bool"));

    return Result::SUCCESS;
}

Result PermutationImpl::create() {
    JST_CHECK(moduleCreate("permutation", permutationModuleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    if (contiguous) {
        JST_CHECK(moduleCreate("duplicate", duplicateModuleConfig, {
            {"buffer", moduleGetOutput({"permutation", "buffer"})}
        }));
        JST_CHECK(moduleExposeOutput("buffer", {"duplicate", "buffer"}));
    } else {
        JST_CHECK(moduleExposeOutput("buffer", {"permutation", "buffer"}));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PermutationImpl,
                   {"permutation"},
                   {"duplicate", true});

}  // namespace Jetstream::Blocks
