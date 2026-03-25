#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct MultiplyImplNativeCpu : public MultiplyImpl,
                               public NativeCpuRuntimeContext,
                               public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelF32();
    Result kernelCF32();

    std::function<Result()> kernel;
};

Result MultiplyImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(MultiplyImpl::create());

    // Register compute kernel.

    if ((a.dtype() == DataType::F32) and
        (b.dtype() == DataType::F32) and
        (c.dtype() == DataType::F32)) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    if ((a.dtype() == DataType::CF32) and
        (b.dtype() == DataType::CF32) and
        (c.dtype() == DataType::CF32)) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_MULTIPLY_NATIVE_CPU] Unsupported data type '{}'.", a.dtype());
    return Result::ERROR;
}

Result MultiplyImplNativeCpu::computeSubmit() {
    return kernel();
}

Result MultiplyImplNativeCpu::kernelF32() {
    return AutomaticIterator<F32, F32, F32>(
        [](const auto& a, const auto& b, auto& c) {
            c = a * b;
        },
    a, b, c);
}

Result MultiplyImplNativeCpu::kernelCF32() {
    return AutomaticIterator<CF32, CF32, CF32>(
        [](const auto& a, const auto& b, auto& c) {
            c = a * b;
        },
    a, b, c);
}

JST_REGISTER_MODULE(MultiplyImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
