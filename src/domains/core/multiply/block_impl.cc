#include <jetstream/domains/core/multiply/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/multiply/module.hh>

namespace Jetstream::Blocks {

struct MultiplyImpl : public Block::Impl, public DynamicConfig<Blocks::Multiply> {
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Multiply> moduleConfig = std::make_shared<Modules::Multiply>();
};

Result MultiplyImpl::define() {
    JST_CHECK(defineInterfaceOutput("product", "Output", "Product of the two input signals."));

    JST_CHECK(defineInterfaceInput("a", "Input A", "First signal to be multiplied."));
    JST_CHECK(defineInterfaceInput("b", "Input B", "Second signal to be multiplied."));

    return Result::SUCCESS;
}

Result MultiplyImpl::create() {
    JST_CHECK(moduleCreate("multiply", moduleConfig, {
        {"a", inputs().at("a")},
        {"b", inputs().at("b")}
    }));
    JST_CHECK(moduleExposeOutput("product", {"multiply", "product"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(MultiplyImpl);

}  // namespace Jetstream::Blocks
