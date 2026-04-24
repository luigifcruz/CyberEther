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

    // Complex integer -> CF32 kernels.
    Result kernelCi8ToCf32();
    Result kernelCi16ToCf32();
    Result kernelCi32ToCf32();
    Result kernelCu8ToCf32();
    Result kernelCu16ToCf32();
    Result kernelCu32ToCf32();

    std::function<Result()> kernel;
};

Result CastImplNativeCpu::create() {
    JST_CHECK(CastImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    // Dispatch kernel based on input/output dtype pair.

    if (outputDtype == DataType::F32) {
        if (input.dtype() == DataType::I8) {
            kernel = [this]() { return kernelI8ToF32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::U8) {
            kernel = [this]() { return kernelU8ToF32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::I16) {
            kernel = [this]() { return kernelI16ToF32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::U16) {
            kernel = [this]() { return kernelU16ToF32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::I32) {
            kernel = [this]() { return kernelI32ToF32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::U32) {
            kernel = [this]() { return kernelU32ToF32(); };
            return Result::SUCCESS;
        }
    }

    if (outputDtype == DataType::CF32) {
        if (input.dtype() == DataType::CI8) {
            kernel = [this]() { return kernelCi8ToCf32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::CI16) {
            kernel = [this]() { return kernelCi16ToCf32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::CI32) {
            kernel = [this]() { return kernelCi32ToCf32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::CU8) {
            kernel = [this]() { return kernelCu8ToCf32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::CU16) {
            kernel = [this]() { return kernelCu16ToCf32(); };
            return Result::SUCCESS;
        }

        if (input.dtype() == DataType::CU32) {
            kernel = [this]() { return kernelCu32ToCf32(); };
            return Result::SUCCESS;
        }
    }

    JST_ERROR("[MODULE_CAST_NATIVE_CPU] Unsupported conversion '{}' -> '{}'.",
              input.dtype(), outputDtype);
    return Result::ERROR;
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
