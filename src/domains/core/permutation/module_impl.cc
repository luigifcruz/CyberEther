#include "module_impl.hh"

namespace Jetstream::Modules {

Result PermutationImpl::validate() {
    const auto& config = *candidate();

    if (config.permutation.empty()) {
        JST_ERROR("[MODULE_PERMUTATION] Permutation cannot be empty.");
        return Result::ERROR;
    }

    std::vector<bool> seen(config.permutation.size(), false);

    for (std::size_t outputAxis = 0; outputAxis < config.permutation.size(); ++outputAxis) {
        const U64 inputAxis = config.permutation[outputAxis];

        if (inputAxis >= config.permutation.size()) {
            JST_ERROR("[MODULE_PERMUTATION] Axis {} is out of range for permutation size {}.",
                      inputAxis, config.permutation.size());
            return Result::ERROR;
        }

        if (seen[inputAxis]) {
            JST_ERROR("[MODULE_PERMUTATION] Axis {} appears more than once.", inputAxis);
            return Result::ERROR;
        }

        seen[inputAxis] = true;
    }

    return Result::SUCCESS;
}

Result PermutationImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result PermutationImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    if (inputTensor.rank() != permutation.size()) {
        JST_ERROR("[MODULE_PERMUTATION] Input tensor rank {} does not match permutation size {}.",
                  inputTensor.rank(), permutation.size());
        return Result::ERROR;
    }

    input = inputTensor;
    output = input.clone();

    JST_CHECK(output.permute(permutation));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
