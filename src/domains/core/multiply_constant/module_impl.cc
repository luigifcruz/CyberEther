#include "module_impl.hh"

namespace Jetstream::Modules {

Result MultiplyConstantImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("factor"));
    JST_CHECK(defineInterfaceOutput("product"));

    return Result::SUCCESS;
}

Result MultiplyConstantImpl::create() {
    const Tensor& inputTensor = inputs().at("factor").tensor;
    input = inputTensor;

    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["product"] = {name(), "product", output};

    return Result::SUCCESS;
}

Result MultiplyConstantImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.constant != constant) {
        constant = config.constant;
        return Result::SUCCESS;
    }

    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
