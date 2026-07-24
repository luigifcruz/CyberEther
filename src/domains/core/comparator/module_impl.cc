#include "module_impl.hh"

#include <cmath>
#include <utility>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr U64 kMaxComparatorInputs = 16;

DataType ErrorDataType(const DataType dtype) {
    switch (dtype) {
        case DataType::F32:
        case DataType::CF32:
            return DataType::F32;
        case DataType::F64:
        case DataType::CF64:
            return DataType::F64;
        default:
            return DataType::None;
    }
}

}  // namespace

std::string ComparatorImpl::inputPortName(const U64 index) {
    return "input" + std::to_string(index);
}

Result ComparatorImpl::validate() {
    validatedInputTensors.clear();
    validatedOutputDevice = DeviceType::None;
    validatedErrorDtype = DataType::None;
    validatedOutputShape.clear();
    validatedOutputSizeBytes = 0;
    validatedInputsReady = false;

    const auto& config = *candidate();

    if (config.inputCount < 2 || config.inputCount > kMaxComparatorInputs) {
        JST_ERROR("[MODULE_COMPARATOR] Input count must be between 2 and {} (got {}).",
                  kMaxComparatorInputs,
                  config.inputCount);
        return Result::ERROR;
    }

    if (!std::isfinite(config.tolerance) || config.tolerance < 0.0) {
        JST_ERROR("[MODULE_COMPARATOR] Tolerance must be finite and non-negative (got {}).",
                  config.tolerance);
        return Result::ERROR;
    }

    const bool topologyChange = !outputs().empty() && config.inputCount != inputCount &&
                                 inputs().size() == inputCount;
    const U64 validationInputCount = topologyChange ? inputCount : config.inputCount;

    for (const auto& [port, _] : inputs()) {
        bool declared = false;
        for (U64 i = 0; i < validationInputCount; ++i) {
            if (port == inputPortName(i)) {
                declared = true;
                break;
            }
        }

        if (!declared) {
            JST_ERROR("[MODULE_COMPARATOR] Received undeclared input '{}'.", port);
            return Result::ERROR;
        }
    }

    if (inputs().size() != validationInputCount) {
        JST_ERROR("[MODULE_COMPARATOR] Expected exactly {} inputs, got {}.",
                  validationInputCount,
                  inputs().size());
        return Result::ERROR;
    }

    std::vector<Tensor> candidateInputTensors;
    candidateInputTensors.reserve(validationInputCount);
    for (U64 i = 0; i < validationInputCount; ++i) {
        const auto port = inputPortName(i);
        if (!inputs().contains(port)) {
            JST_ERROR("[MODULE_COMPARATOR] Missing input '{}'.", port);
            return Result::ERROR;
        }
        candidateInputTensors.push_back(inputs().at(port).tensor);
    }

    const Tensor& reference = candidateInputTensors.front();
    for (const Tensor& tensor : candidateInputTensors) {
        if (!tensor.validShape() || tensor.size() == 0) {
            return Result::SUCCESS;
        }
    }

    const Shape& referenceShape = reference.shape();
    const DataType referenceDtype = reference.dtype();
    const DeviceType referenceDevice = reference.device();

    if (referenceShape.empty()) {
        JST_ERROR("[MODULE_COMPARATOR] Rank-zero inputs are not supported.");
        return Result::ERROR;
    }

    for (U64 i = 1; i < candidateInputTensors.size(); ++i) {
        const Tensor& tensor = candidateInputTensors[i];

        if (tensor.shape() != referenceShape) {
            JST_ERROR("[MODULE_COMPARATOR] Input {} shape {} does not match reference shape {}.",
                      i,
                      tensor.shape(),
                      referenceShape);
            return Result::ERROR;
        }

        if (tensor.dtype() != referenceDtype) {
            JST_ERROR("[MODULE_COMPARATOR] Input {} dtype {} does not match reference dtype {}.",
                      i,
                      tensor.dtype(),
                      referenceDtype);
            return Result::ERROR;
        }

        if (tensor.device() != referenceDevice) {
            JST_ERROR("[MODULE_COMPARATOR] Input {} device {} does not match reference device {}.",
                      i,
                      tensor.device(),
                      referenceDevice);
            return Result::ERROR;
        }
    }

    const DataType errorDtype = ErrorDataType(referenceDtype);
    U64 outputSizeBytes = 0;
    if (errorDtype != DataType::None &&
        !detail::CheckedMultiply(reference.size(),
                                 static_cast<U64>(DataTypeSize(errorDtype)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_COMPARATOR] Error output exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedOutputDevice = referenceDevice;
    validatedErrorDtype = errorDtype;
    validatedOutputShape = referenceShape;
    validatedOutputSizeBytes = outputSizeBytes;
    validatedInputTensors = std::move(candidateInputTensors);
    validatedInputsReady = true;

    return Result::SUCCESS;
}

Result ComparatorImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    for (U64 i = 0; i < inputCount; ++i) {
        JST_CHECK(defineInterfaceInput(inputPortName(i)));
    }

    JST_CHECK(defineInterfaceOutput("error"));

    return Result::SUCCESS;
}

Result ComparatorImpl::create() {
    inputTensors = validatedInputTensors;

    JST_CHECK(error.create(validatedOutputDevice, validatedErrorDtype, validatedOutputShape));
    error.propagateAttributes(inputTensors.front());

    outputs()["error"].produced(name(), "error", error);

    maxDiffState.publish(0.0);
    meanDiffState.publish(0.0);
    mseState.publish(0.0);
    matchState.publish(true);

    return Result::SUCCESS;
}

Result ComparatorImpl::destroy() {
    maxDiffState.publish(0.0);
    meanDiffState.publish(0.0);
    mseState.publish(0.0);
    matchState.publish(true);
    return Result::SUCCESS;
}

Result ComparatorImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.inputCount != inputCount) {
        return Result::RECREATE;
    }

    tolerance = config.tolerance;

    return Result::SUCCESS;
}

F64 ComparatorImpl::getMaxDiff() const {
    return maxDiffState.get();
}

F64 ComparatorImpl::getMeanDiff() const {
    return meanDiffState.get();
}

F64 ComparatorImpl::getMse() const {
    return mseState.get();
}

bool ComparatorImpl::getMatch() const {
    return matchState.get();
}

}  // namespace Jetstream::Modules
