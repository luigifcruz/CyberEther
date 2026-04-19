#include "module_impl.hh"

namespace Jetstream::Modules {

Result CastImpl::validate() {
    const auto& config = *candidate();

    const DataType outDtype = NameToDataType(config.outputType);
    if (outDtype == DataType::None) {
        JST_ERROR("[MODULE_CAST] Invalid output type '{}'.", config.outputType);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result CastImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result CastImpl::create() {
    // Get input and output dtype.

    input = inputs().at("buffer").tensor;
    outputDtype = NameToDataType(outputType);

    // Configure default scaler based on input type.

    switch (input.dtype()) {
        case DataType::F32:
            scaler = 1.0f;
            break;
        case DataType::I8:
        case DataType::CI8:
        case DataType::U8:
        case DataType::CU8:
            scaler = 128.0f;
            break;
        case DataType::I16:
        case DataType::CI16:
        case DataType::U16:
        case DataType::CU16:
            scaler = 32768.0f;
            break;
        case DataType::I32:
        case DataType::CI32:
        case DataType::U32:
        case DataType::CU32:
            scaler = 2147483648.0f;
            break;
        default:
            JST_ERROR("[MODULE_CAST] No default scaler for input type '{}'.",
                      input.dtype());
            return Result::ERROR;
    }

    // Allocate output with configured type and same shape.

    JST_CHECK(output.create(input.device(), outputDtype, input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
