#include "module_impl.hh"

#include <limits>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result FftImpl::validate() {
    validatedResolvedAxis = 0;
    validatedOutputShape.clear();
    validatedOutputElementCount = 0;
    validatedOutputSizeBytes = 0;

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() == 0 ||
        inputTensor.rank() > static_cast<U64>(std::numeric_limits<I64>::max())) {
        JST_ERROR("[MODULE_FFT] Expected an input tensor with at least one dimension.");
        return Result::ERROR;
    }

    const auto& config = *candidate();
    const auto candidateAxis = ResolveAxis(config.axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_FFT] Axis {} is out of bounds for a rank-{} tensor.",
                  config.axis,
                  inputTensor.rank());
        return Result::ERROR;
    }

    U64 outputElementCount = 1;
    for (const U64 dimension : inputTensor.shape()) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_FFT] Output shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(inputTensor.dtype())),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_FFT] Output shape exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedResolvedAxis = *candidateAxis;
    validatedOutputShape = inputTensor.shape();
    validatedOutputElementCount = outputElementCount;
    validatedOutputSizeBytes = outputSizeBytes;

    return Result::SUCCESS;
}

Result FftImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result FftImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;
    resolvedAxis = validatedResolvedAxis;

    JST_CHECK(output.create(input.device(), input.dtype(), validatedOutputShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result FftImpl::destroy() {
    return Result::SUCCESS;
}

Result FftImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
