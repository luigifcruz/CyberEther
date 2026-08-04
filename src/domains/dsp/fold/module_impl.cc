#include "module_impl.hh"

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result FoldImpl::validate() {
    validatedResolvedAxis = 0;
    validatedChannelAxis.reset();
    validatedChannelOffsets.clear();
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

    std::vector<U64> channelOffsets;
    if (inputTensor.hasAttribute("channelOffsets")) {
        const std::any value = inputTensor.attribute("channelOffsets");
        const auto* typedOffsets = std::any_cast<std::vector<U64>>(&value);
        if (typedOffsets == nullptr) {
            JST_ERROR("[MODULE_FOLD] Input channelOffsets metadata must have "
                      "type vector<U64>.");
            return Result::ERROR;
        }
        if (typedOffsets->empty()) {
            JST_ERROR("[MODULE_FOLD] Input channelOffsets metadata cannot be empty.");
            return Result::ERROR;
        }
        channelOffsets = *typedOffsets;
    }

    if (channelOffsets.empty()) {
        if (axisSize < config.offset) {
            JST_ERROR("[MODULE_FOLD] Offset ({}) is greater than the "
                      "input shape ({}) along axis ({}).",
                      config.offset, axisSize, *axes.sample);
            return Result::ERROR;
        }
    } else {
        if (!axes.channel ||
            channelOffsets.size() != inputTensor.shape(*axes.channel)) {
            JST_ERROR("[MODULE_FOLD] Channel offsets must match channelAxis extent.");
            return Result::ERROR;
        }
        for (U64 channel = 0; channel < channelOffsets.size(); ++channel) {
            if (axisSize < channelOffsets[channel]) {
                JST_ERROR("[MODULE_FOLD] Channel offset #{} ({}) is greater than "
                          "the input shape ({}) along axis ({}).",
                          channel,
                          channelOffsets[channel],
                          axisSize,
                          *axes.sample);
                return Result::ERROR;
            }
        }
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
    if (!channelOffsets.empty()) {
        validatedChannelAxis = axes.channel;
        validatedChannelOffsets = channelOffsets;
    }
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
    channelAxis = validatedChannelAxis;
    channelOffsets = validatedChannelOffsets;
    decimationFactor = validatedDecimationFactor;

    // Build output shape.
    auto outputShape = input.shape();
    outputShape[resolvedAxis] = size;

    // Allocate output tensor with same dtype.
    JST_CHECK(output.create(input.device(), input.dtype(), outputShape));
    JST_CHECK(output.propagateAttributes(input));
    JST_CHECK(output.removeAttribute("channelOffsets"));

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
