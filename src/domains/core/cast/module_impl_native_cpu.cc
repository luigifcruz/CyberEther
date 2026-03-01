#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct CastImplNativeCpu : public CastImpl,
                           public Runtime::Context,
                           public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
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

    // Dispatch kernel based on input/output dtype pair.

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
    return kernel();
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
