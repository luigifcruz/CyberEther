#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct MultiplyConstantImplNativeCpu : public MultiplyConstantImpl,
                                       public NativeCpuRuntimeContext,
                                       public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelF32();
    Result kernelCF32();
    Result kernelF64();
    Result kernelCF64();

    std::function<Result()> kernel;
};

Result MultiplyConstantImplNativeCpu::create() {
    JST_CHECK(MultiplyConstantImpl::create());

    if (input.dtype() == DataType::F32 && output.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::CF32 && output.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }
    if (input.dtype() == DataType::F64 && output.dtype() == DataType::F64) {
        kernel = [this]() { return kernelF64(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::CF64 && output.dtype() == DataType::CF64) {
        kernel = [this]() { return kernelCF64(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_MULTIPLY_CONSTANT_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result MultiplyConstantImplNativeCpu::computeSubmit() {
    return kernel();
}

Result MultiplyConstantImplNativeCpu::kernelF32() {
    const F32 c = constant;

    return AutomaticIterator<F32, F32>(
        [c](const auto& in, auto& out) {
            out = in * c;
        },
    input, output);
}

Result MultiplyConstantImplNativeCpu::kernelCF32() {
    const F32 c = constant;

    return AutomaticIterator<CF32, CF32>(
        [c](const auto& in, auto& out) {
            out = in * c;
        },
    input, output);
}

Result MultiplyConstantImplNativeCpu::kernelF64() {
    const F64 c = constant;

    return AutomaticIterator<F64, F64>(
        [c](const auto& in, auto& out) {
            out = in * c;
        },
    input, output);
}

Result MultiplyConstantImplNativeCpu::kernelCF64() {
    const F64 c = constant;

    return AutomaticIterator<CF64, CF64>(
        [c](const auto& in, auto& out) {
            out = in * c;
        },
    input, output);
}

JST_REGISTER_MODULE(MultiplyConstantImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
