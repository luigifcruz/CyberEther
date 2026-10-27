#include "module_impl.hh"

#include <cmath>
#include <limits>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

template<typename T>
CastImpl::Scale IntegerScale() {
    using Limits = std::numeric_limits<T>;

    CastImpl::Scale scale;
    scale.integer = true;
    scale.isUnsigned = !Limits::is_signed;
    scale.bits = Limits::digits + Limits::is_signed;
    scale.fullScale = std::ldexp(1.0, static_cast<int>(scale.bits) - 1);
    return scale;
}

}  // namespace

CastImpl::Scale CastImpl::ScaleOf(const DataType& dtype) {
    switch (dtype) {
        case DataType::I8:
        case DataType::CI8:
            return IntegerScale<I8>();
        case DataType::I16:
        case DataType::CI16:
            return IntegerScale<I16>();
        case DataType::I32:
        case DataType::CI32:
            return IntegerScale<I32>();
        case DataType::I64:
        case DataType::CI64:
            return IntegerScale<I64>();
        case DataType::U8:
        case DataType::CU8:
            return IntegerScale<U8>();
        case DataType::U16:
        case DataType::CU16:
            return IntegerScale<U16>();
        case DataType::U32:
        case DataType::CU32:
            return IntegerScale<U32>();
        case DataType::U64:
        case DataType::CU64:
            return IntegerScale<U64>();
        default:
            return {};
    }
}

Result CastImpl::validate() {
    const auto& config = *candidate();

    const DataType candidateOutputDtype = NameToDataType(config.outputType);
    if (candidateOutputDtype == DataType::None) {
        JST_ERROR("[MODULE_CAST] Invalid output type '{}'.", config.outputType);
        return Result::ERROR;
    }

    bool candidateBypass = false;
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
        }

        if (!candidateBypass && IsDataTypeComplex(candidateInput.dtype()) &&
            !IsDataTypeComplex(candidateOutputDtype)) {
            JST_ERROR("[MODULE_CAST] Cannot cast complex '{}' to real '{}'. Use a "
                      "Complex Split to pick the parts.",
                      candidateInput.dtype(), candidateOutputDtype);
            return Result::ERROR;
        }
    }

    validatedOutputDtype = candidateOutputDtype;
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
    inputScale = ScaleOf(input.dtype());
    outputScale = ScaleOf(outputDtype);
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
