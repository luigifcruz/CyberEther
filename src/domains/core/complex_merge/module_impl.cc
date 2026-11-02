#include "module_impl.hh"

namespace Jetstream::Modules {

Result ComplexMergeImpl::validate() {
    const auto& config = *candidate();

    if (config.mode != "rectangular" && config.mode != "polar") {
        JST_ERROR("[MODULE_COMPLEX_MERGE] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (inputs().contains("a") && inputs().contains("b")) {
        const Tensor& candidateA = inputs().at("a").tensor;
        const Tensor& candidateB = inputs().at("b").tensor;

        if (candidateA.validShape() && candidateB.validShape() &&
            candidateA.size() != 0 && candidateB.size() != 0) {
            if (candidateA.dtype() != DataType::F32 ||
                candidateB.dtype() != DataType::F32) {
                JST_ERROR("[MODULE_COMPLEX_MERGE] Unsupported input data types '{}' and '{}'.",
                          candidateA.dtype(), candidateB.dtype());
                return Result::ERROR;
            }

            if (candidateA.shape() != candidateB.shape()) {
                JST_ERROR("[MODULE_COMPLEX_MERGE] Input shapes {} and {} do not match.",
                          candidateA.shape(), candidateB.shape());
                return Result::ERROR;
            }

            if (candidateA.shape().empty()) {
                JST_ERROR("[MODULE_COMPLEX_MERGE] Cannot allocate a rank-zero output.");
                return Result::ERROR;
            }
        }
    }

    validatedPolar = config.mode == "polar";
    return Result::SUCCESS;
}

Result ComplexMergeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("a"));
    JST_CHECK(defineInterfaceInput("b"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ComplexMergeImpl::create() {
    inputA = inputs().at("a").tensor;
    inputB = inputs().at("b").tensor;
    polar = validatedPolar;

    JST_CHECK(output.create(inputA.device(), DataType::CF32, inputA.shape()));
    JST_CHECK(output.propagateAttributes(inputA));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
