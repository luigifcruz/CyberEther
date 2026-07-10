#include "module_impl.hh"

#include <cmath>

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
    inputTensors.clear();
    inputTensors.reserve(inputCount);

    for (U64 i = 0; i < inputCount; ++i) {
        inputTensors.push_back(inputs().at(inputPortName(i)).tensor);
    }

    const Tensor& reference = inputTensors.front();
    const Shape& referenceShape = reference.shape();
    const DataType referenceDtype = reference.dtype();
    const DeviceType device = reference.device();

    const DataType errorDtype = ErrorDataType(referenceDtype);
    if (errorDtype == DataType::None) {
        JST_ERROR("[MODULE_COMPARATOR] Unsupported data type '{}'.", referenceDtype);
        return Result::ERROR;
    }

    for (U64 i = 1; i < inputTensors.size(); ++i) {
        const Tensor& tensor = inputTensors[i];

        if (tensor.dtype() != referenceDtype) {
            JST_ERROR("[MODULE_COMPARATOR] Input {} dtype {} does not match reference dtype {}.",
                      i,
                      tensor.dtype(),
                      referenceDtype);
            return Result::ERROR;
        }

        if (tensor.device() != device) {
            JST_ERROR("[MODULE_COMPARATOR] Input {} device {} does not match reference device {}.",
                      i,
                      tensor.device(),
                      device);
            return Result::ERROR;
        }

        if (tensor.shape() != referenceShape) {
            JST_ERROR("[MODULE_COMPARATOR] Input {} shape {} does not match reference shape {}.",
                      i,
                      tensor.shape(),
                      referenceShape);
            return Result::ERROR;
        }
    }

    JST_CHECK(error.create(device, errorDtype, referenceShape));
    error.propagateAttributes(reference);

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
