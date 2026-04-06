#include "module_impl.hh"

#include <algorithm>

namespace Jetstream::Modules {

namespace {

template<typename T>
void FillTensor(Tensor& tensor, const T& value) {
    T* outputData = tensor.data<T>();
    std::fill(outputData, outputData + tensor.size(), value);
}

}  // namespace

Result OnesTensorImpl::validate() {
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

    if (config.dataType != "F32" && config.dataType != "CF32") {
        JST_ERROR("[MODULE_ONES_TENSOR] Invalid data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result OnesTensorImpl::define() {
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result OnesTensorImpl::create() {
    Buffer::Config outputConfig{};
    outputConfig.hostAccessible = true;

    JST_CHECK(output.create(device(), NameToDataType(dataType), shape, outputConfig));

    if (output.dtype() == DataType::F32) {
        FillTensor(output, 1.0f);
    } else if (output.dtype() == DataType::CF32) {
        FillTensor(output, CF32(1.0f, 0.0f));
    } else {
        JST_ERROR("[MODULE_ONES_TENSOR] Unsupported data type '{}'.", output.dtype());
        return Result::ERROR;
    }

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
