#include "module_impl.hh"

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result FftImpl::validate() {
    validatedResolvedAxis = 0;
    validatedOutputShape.clear();
    validatedOutputDataType = DataType::None;
    validatedOutputElementCount = 0;
    validatedOutputSizeBytes = 0;

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto& config = *candidate();
    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_FFT] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    validatedOutputDataType = inputTensor.dtype();
    validatedOutputShape = inputTensor.shape();
    if (inputTensor.dtype() == DataType::F32 && config.forward &&
        config.complexOutput) {
        validatedOutputDataType = DataType::CF32;
        validatedOutputShape[*axes.sample] =
            (inputTensor.shape(*axes.sample) / 2) + 1;
    }

    U64 outputElementCount = 1;
    for (const U64 dimension : validatedOutputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_FFT] Output shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(validatedOutputDataType)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_FFT] Output shape exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedResolvedAxis = *axes.sample;
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

    JST_CHECK(output.create(input.device(), validatedOutputDataType, validatedOutputShape));
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
