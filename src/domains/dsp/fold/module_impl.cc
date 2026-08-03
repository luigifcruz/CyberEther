#include "module_impl.hh"

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result FoldImpl::validate() {
    validatedResolvedAxis = 0;
    validatedDecimationFactor = 0;
    validatedOutputSizeBytes = 0;

    const auto& config = *candidate();

    if (config.size == 0) {
        JST_ERROR("[MODULE_FOLD] Size cannot be zero.");
        return Result::ERROR;
    }

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_FOLD] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    if (inputTensor.hasAttribute("sampleRate")) {
        const std::any value = inputTensor.attribute("sampleRate");
        if (!std::any_cast<F32>(&value)) {
            JST_ERROR("[MODULE_FOLD] Input sample rate metadata must have type F32.");
            return Result::ERROR;
        }
    }

    const U64 axisSize = inputTensor.shape(*axes.sample);
    if (axisSize % config.size != 0) {
        JST_ERROR("[MODULE_FOLD] Size ({}) is not a divisor of "
                  "the input shape ({}) along axis ({}).",
                  config.size, axisSize, *axes.sample);
        return Result::ERROR;
    }

    if (axisSize < config.offset) {
        JST_ERROR("[MODULE_FOLD] Offset ({}) is greater than the "
                  "input shape ({}) along axis ({}).",
                  config.offset, axisSize, *axes.sample);
        return Result::ERROR;
    }

    const U64 decimationFactor = axisSize / config.size;
    const U64 outputElementCount = inputTensor.size() / decimationFactor;
    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(inputTensor.dtype())),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_FOLD] Output exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedResolvedAxis = *axes.sample;
    validatedDecimationFactor = decimationFactor;
    validatedOutputSizeBytes = outputSizeBytes;
    return Result::SUCCESS;
}

Result FoldImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result FoldImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;
    resolvedAxis = validatedResolvedAxis;
    decimationFactor = validatedDecimationFactor;

    // Build output shape.
    auto outputShape = input.shape();
    outputShape[resolvedAxis] = size;

    // Allocate output tensor with same dtype.
    JST_CHECK(output.create(input.device(), input.dtype(), outputShape));
    JST_CHECK(output.propagateAttributes(input));

    if (input.hasAttribute("sampleRate")) {
        const Tensor inputCopy = input;
        const F32 foldDecimation = static_cast<F32>(decimationFactor);
        JST_CHECK(output.setDerivedAttribute(
            "sampleRate",
            [inputCopy, foldDecimation]() -> std::any {
                const std::any sampleRate = inputCopy.attribute("sampleRate");
                const auto* sampleRateF32 = std::any_cast<F32>(&sampleRate);
                if (sampleRateF32 == nullptr) {
                    return {};
                }
                return std::any(*sampleRateF32 / foldDecimation);
            }));
    }

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
