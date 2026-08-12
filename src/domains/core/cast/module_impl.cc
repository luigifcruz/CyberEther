#include "module_impl.hh"

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result CastImpl::validate() {
    const auto& config = *candidate();

    const DataType candidateOutputDtype = NameToDataType(config.outputType);
    if (candidateOutputDtype == DataType::None) {
        JST_ERROR("[MODULE_CAST] Invalid output type '{}'.", config.outputType);
        return Result::ERROR;
    }

    bool candidateBypass = false;
    F32 candidateScaler = 1.0f;
    U64 candidateOutputElementCount = 0;
    U64 candidateOutputSizeBytes = 0;

    if (inputs().contains("buffer")) {
        const Tensor& candidateInput = inputs().at("buffer").tensor;
        candidateBypass = candidateInput.dtype() == candidateOutputDtype;

        if (candidateInput.validShape() && candidateInput.size() != 0 &&
            !candidateBypass) {
            if (candidateInput.shape().empty()) {
                JST_ERROR("[MODULE_CAST] Cannot allocate a rank-zero cast output.");
                return Result::ERROR;
            }

            candidateOutputElementCount = 1;
            for (const U64 dimension : candidateInput.shape()) {
                if (!detail::CheckedMultiply(candidateOutputElementCount,
                                             dimension,
                                             candidateOutputElementCount)) {
                    JST_ERROR("[MODULE_CAST] Output exceeds the supported layout range.");
                    return Result::ERROR;
                }
            }

            if (!detail::CheckedMultiply(
                    candidateOutputElementCount,
                    static_cast<U64>(DataTypeSize(candidateOutputDtype)),
                    candidateOutputSizeBytes)) {
                JST_ERROR("[MODULE_CAST] Output exceeds the supported byte range.");
                return Result::ERROR;
            }

            switch (candidateInput.dtype()) {
                case DataType::I8:
                case DataType::CI8:
                case DataType::U8:
                case DataType::CU8:
                    candidateScaler = 128.0f;
                    break;
                case DataType::I16:
                case DataType::CI16:
                case DataType::U16:
                case DataType::CU16:
                    candidateScaler = 32768.0f;
                    break;
                case DataType::I32:
                case DataType::CI32:
                case DataType::U32:
                case DataType::CU32:
                    candidateScaler = 2147483648.0f;
                    break;
                default:
                    break;
            }
        }
    }

    validatedOutputDtype = candidateOutputDtype;
    validatedScaler = candidateScaler;
    validatedBypass = candidateBypass;
    validatedOutputElementCount = candidateOutputElementCount;
    validatedOutputSizeBytes = candidateOutputSizeBytes;
    return Result::SUCCESS;
}

Result CastImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result CastImpl::create() {
    input = inputs().at("buffer").tensor;
    outputDtype = validatedOutputDtype;
    scaler = validatedScaler;
    bypass = validatedBypass;

    if (bypass) {
        outputs()["buffer"].produced(name(), "buffer", input);
        return Result::SUCCESS;
    }

    // Allocate output with configured type and same shape.

    JST_CHECK(output.create(input.device(), outputDtype, input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
