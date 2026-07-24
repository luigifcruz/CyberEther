#include "module_impl.hh"

#include <algorithm>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

template<typename T>
void FillTensor(Tensor& tensor, const T& value) {
    T* outputData = tensor.data<T>();
    std::fill(outputData, outputData + tensor.size(), value);
}

}  // namespace

Result OnesTensorImpl::validate() {
    validatedDataType = DataType::None;
    validatedElementCount = 0;
    validatedOutputSizeBytes = 0;

    const auto& config = *candidate();

    if (config.shape.empty()) {
        JST_ERROR("[MODULE_ONES_TENSOR] Shape cannot be empty.");
        return Result::ERROR;
    }

    for (Index axis = 0; axis < config.shape.size(); ++axis) {
        if (config.shape[axis] == 0) {
            JST_ERROR("[MODULE_ONES_TENSOR] Shape dimension {} cannot be zero.", axis);
            return Result::ERROR;
        }
    }

    if (config.dataType != "F32" && config.dataType != "CF32" &&
        config.dataType != "F64" && config.dataType != "CF64") {
        JST_ERROR("[MODULE_ONES_TENSOR] Invalid data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    validatedDataType = NameToDataType(config.dataType);
    U64 elementCount = 1;
    for (const U64 dimension : config.shape) {
        if (!detail::CheckedMultiply(elementCount, dimension, elementCount)) {
            JST_ERROR("[MODULE_ONES_TENSOR] Shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(elementCount,
                                 static_cast<U64>(DataTypeSize(validatedDataType)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_ONES_TENSOR] Tensor exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedElementCount = elementCount;
    validatedOutputSizeBytes = outputSizeBytes;
    return Result::SUCCESS;
}

Result OnesTensorImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATIC_OUTPUT));

    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result OnesTensorImpl::create() {
    Buffer::Config outputConfig{};
    outputConfig.hostAccessible = true;

    JST_CHECK(output.create(device(), validatedDataType, shape, outputConfig));
    JST_CHECK(fillOutput());

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

Result OnesTensorImpl::fillOutput() {
    if (output.dtype() == DataType::F32) {
        FillTensor(output, 1.0f);
    } else if (output.dtype() == DataType::CF32) {
        FillTensor(output, CF32(1.0f, 0.0f));
    } else if (output.dtype() == DataType::F64) {
        FillTensor(output, 1.0);
    } else if (output.dtype() == DataType::CF64) {
        FillTensor(output, CF64(1.0, 0.0));
    } else {
        JST_ERROR("[MODULE_ONES_TENSOR] Unsupported data type '{}'.", output.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
