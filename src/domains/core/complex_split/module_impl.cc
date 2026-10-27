#include "module_impl.hh"

namespace Jetstream::Modules {

Result ComplexSplitImpl::validate() {
    const auto& config = *candidate();

    if (config.mode != "rectangular" && config.mode != "polar") {
        JST_ERROR("[MODULE_COMPLEX_SPLIT] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (inputs().contains("buffer")) {
        const Tensor& candidateInput = inputs().at("buffer").tensor;
        if (candidateInput.validShape() && candidateInput.size() != 0) {
            if (candidateInput.dtype() != DataType::CF32) {
                JST_ERROR("[MODULE_COMPLEX_SPLIT] Unsupported input data type '{}'.",
                          candidateInput.dtype());
                return Result::ERROR;
            }

            if (candidateInput.shape().empty()) {
                JST_ERROR("[MODULE_COMPLEX_SPLIT] Cannot allocate rank-zero outputs.");
                return Result::ERROR;
            }
        }
    }

    validatedPolar = config.mode == "polar";
    return Result::SUCCESS;
}

Result ComplexSplitImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("a"));
    JST_CHECK(defineInterfaceOutput("b"));

    return Result::SUCCESS;
}

Result ComplexSplitImpl::create() {
    input = inputs().at("buffer").tensor;
    polar = validatedPolar;

    JST_CHECK(outputA.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(outputB.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(outputA.propagateAttributes(input));
    JST_CHECK(outputB.propagateAttributes(input));

    outputs()["a"].produced(name(), "a", outputA);
    outputs()["b"].produced(name(), "b", outputB);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
