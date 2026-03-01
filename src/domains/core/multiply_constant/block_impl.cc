#include <jetstream/domains/core/multiply_constant/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/multiply_constant/module.hh>

namespace Jetstream::Blocks {

struct MultiplyConstantImpl : public Block::Impl, public DynamicConfig<Blocks::MultiplyConstant> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::MultiplyConstant> moduleConfig =
        std::make_shared<Modules::MultiplyConstant>();
};

Result MultiplyConstantImpl::configure() {
    moduleConfig->constant = constant;

    return Result::SUCCESS;
}

Result MultiplyConstantImpl::define() {
    JST_CHECK(defineInterfaceInput("factor", "Input", "Input signal to multiply."));
    JST_CHECK(defineInterfaceOutput("product", "Output", "Product of input and constant."));

    JST_CHECK(defineInterfaceConfig("constant",
                                    "Constant",
                                    "Scalar value to multiply each element by.",
                                    "float::3"));

    return Result::SUCCESS;
}

Result MultiplyConstantImpl::create() {
    JST_CHECK(moduleCreate("multiply_constant", moduleConfig, {
        {"factor", inputs().at("factor")}
    }));
    JST_CHECK(moduleExposeOutput("product", {"multiply_constant", "product"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(MultiplyConstantImpl);

}  // namespace Jetstream::Blocks
