#include <limits>

#include <jetstream/memory/macros.hh>
#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct CastImplNativeCpu : public CastImpl,
                           public NativeCpuRuntimeContext,
                           public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    // Real -> F32 kernels.
    Result kernelI8ToF32();
    Result kernelU8ToF32();
    Result kernelI16ToF32();
    Result kernelU16ToF32();
    Result kernelI32ToF32();
    Result kernelU32ToF32();

    // Real floating-point -> complex floating-point kernel.
    Result kernelF32ToCf32();

    // Complex integer -> CF32 kernels.
    Result kernelCi8ToCf32();
    Result kernelCi16ToCf32();
    Result kernelCi32ToCf32();
    Result kernelCu8ToCf32();
    Result kernelCu16ToCf32();
    Result kernelCu32ToCf32();

    std::function<Result()> kernel;
};

Result CastImplNativeCpu::validate() {
    JST_CHECK(CastImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& candidateInput = inputs().at("buffer").tensor;
    if (!candidateInput.validShape() || candidateInput.size() == 0) {
        return Result::SUCCESS;
    }

    if (validatedBypass) {
        return Result::SUCCESS;
    }

    const DataType inputDtype = candidateInput.dtype();
    const bool supportedPair =
        (validatedOutputDtype == DataType::F32 &&
         (inputDtype == DataType::I8 || inputDtype == DataType::U8 ||
          inputDtype == DataType::I16 || inputDtype == DataType::U16 ||
          inputDtype == DataType::I32 || inputDtype == DataType::U32)) ||
        (validatedOutputDtype == DataType::CF32 &&
         (inputDtype == DataType::F32 ||
          inputDtype == DataType::CI8 || inputDtype == DataType::CU8 ||
          inputDtype == DataType::CI16 || inputDtype == DataType::CU16 ||
          inputDtype == DataType::CI32 || inputDtype == DataType::CU32));
    if (!supportedPair) {
        JST_ERROR("[MODULE_CAST_NATIVE_CPU] Unsupported conversion '{}' -> '{}'.",
                  inputDtype, validatedOutputDtype);
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                        alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_CAST_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result CastImplNativeCpu::create() {
    JST_CHECK(CastImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    switch (input.dtype()) {
        case DataType::I8:
            kernel = [this]() { return kernelI8ToF32(); };
            break;
        case DataType::U8:
            kernel = [this]() { return kernelU8ToF32(); };
            break;
        case DataType::I16:
            kernel = [this]() { return kernelI16ToF32(); };
            break;
        case DataType::U16:
            kernel = [this]() { return kernelU16ToF32(); };
            break;
        case DataType::I32:
            kernel = [this]() { return kernelI32ToF32(); };
            break;
        case DataType::U32:
            kernel = [this]() { return kernelU32ToF32(); };
            break;
        case DataType::F32:
            kernel = [this]() { return kernelF32ToCf32(); };
            break;
        case DataType::CI8:
            kernel = [this]() { return kernelCi8ToCf32(); };
            break;
        case DataType::CI16:
            kernel = [this]() { return kernelCi16ToCf32(); };
            break;
        case DataType::CI32:
            kernel = [this]() { return kernelCi32ToCf32(); };
            break;
        case DataType::CU8:
            kernel = [this]() { return kernelCu8ToCf32(); };
            break;
        case DataType::CU16:
            kernel = [this]() { return kernelCu16ToCf32(); };
            break;
        case DataType::CU32:
            kernel = [this]() { return kernelCu32ToCf32(); };
            break;
        default:
            break;
    }

    return Result::SUCCESS;
}

Result CastImplNativeCpu::computeSubmit() {
    if (bypass) {
        return Result::SUCCESS;
    }

    return kernel();
}

Result CastImplNativeCpu::kernelI8ToF32() {
    const F32 s = scaler;

    return AutomaticIterator<const I8, F32>(
        [s](const auto& in, auto& out) {
            out = static_cast<F32>(in) / s;
        },
    input, output);
}

Result CastImplNativeCpu::kernelU8ToF32() {
    const F32 s = scaler;

    return AutomaticIterator<const U8, F32>(
        [s](const auto& in, auto& out) {
            out = static_cast<F32>(in) / s;
        },
    input, output);
}

Result CastImplNativeCpu::kernelI16ToF32() {
    const F32 s = scaler;

    return AutomaticIterator<const I16, F32>(
        [s](const auto& in, auto& out) {
            out = static_cast<F32>(in) / s;
        },
    input, output);
}

Result CastImplNativeCpu::kernelU16ToF32() {
    const F32 s = scaler;

    return AutomaticIterator<const U16, F32>(
        [s](const auto& in, auto& out) {
            out = static_cast<F32>(in) / s;
        },
    input, output);
}

Result CastImplNativeCpu::kernelI32ToF32() {
    const F32 s = scaler;

    return AutomaticIterator<const I32, F32>(
        [s](const auto& in, auto& out) {
            out = static_cast<F32>(in) / s;
        },
    input, output);
}

Result CastImplNativeCpu::kernelU32ToF32() {
    const F32 s = scaler;

    return AutomaticIterator<const U32, F32>(
        [s](const auto& in, auto& out) {
            out = static_cast<F32>(in) / s;
        },
    input, output);
}

Result CastImplNativeCpu::kernelF32ToCf32() {
    return AutomaticIterator<const F32, CF32>(
        [](const auto& in, auto& out) {
            out = CF32(in, 0.0f);
        },
    input, output);
}

Result CastImplNativeCpu::kernelCi8ToCf32() {
    const F32 s = scaler;

    return AutomaticIterator<const CI8, CF32>(
        [s](const auto& in, auto& out) {
            out = CF32(static_cast<F32>(in.real()) / s,
                       static_cast<F32>(in.imag()) / s);
        },
    input, output);
}

Result CastImplNativeCpu::kernelCi16ToCf32() {
    const F32 s = scaler;

    return AutomaticIterator<const CI16, CF32>(
        [s](const auto& in, auto& out) {
            out = CF32(static_cast<F32>(in.real()) / s,
                       static_cast<F32>(in.imag()) / s);
        },
    input, output);
}

Result CastImplNativeCpu::kernelCi32ToCf32() {
    const F32 s = scaler;

    return AutomaticIterator<const CI32, CF32>(
        [s](const auto& in, auto& out) {
            out = CF32(static_cast<F32>(in.real()) / s,
                       static_cast<F32>(in.imag()) / s);
        },
    input, output);
}

Result CastImplNativeCpu::kernelCu8ToCf32() {
    const F32 s = scaler;

    return AutomaticIterator<const CU8, CF32>(
        [s](const auto& in, auto& out) {
            out = CF32(static_cast<F32>(in.real()) / s,
                       static_cast<F32>(in.imag()) / s);
        },
    input, output);
}

Result CastImplNativeCpu::kernelCu16ToCf32() {
    const F32 s = scaler;

    return AutomaticIterator<const CU16, CF32>(
        [s](const auto& in, auto& out) {
            out = CF32(static_cast<F32>(in.real()) / s,
                       static_cast<F32>(in.imag()) / s);
        },
    input, output);
}

Result CastImplNativeCpu::kernelCu32ToCf32() {
    const F32 s = scaler;

    return AutomaticIterator<const CU32, CF32>(
        [s](const auto& in, auto& out) {
            out = CF32(static_cast<F32>(in.real()) / s,
                       static_cast<F32>(in.imag()) / s);
        },
    input, output);
}

JST_REGISTER_MODULE(CastImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
