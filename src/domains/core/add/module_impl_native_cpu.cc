#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct AddImplNativeCpu : public AddImpl,
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

Result AddImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(AddImpl::create());

    // Register compute kernel.

    if ((a.dtype() == DataType::F32) &&
        (b.dtype() == DataType::F32) &&
        (c.dtype() == DataType::F32)) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    if ((a.dtype() == DataType::CF32) &&
        (b.dtype() == DataType::CF32) &&
        (c.dtype() == DataType::CF32)) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_ADD_NATIVE_CPU] Unsupported data type '{}'.", a.dtype());
    return Result::ERROR;
}

Result AddImplNativeCpu::computeSubmit() {
    return kernel();
}

Result AddImplNativeCpu::kernelF32() {
    return AutomaticIterator<F32, F32, F32>(
        [](const auto& a, const auto& b, auto& c) {
            c = a + b;
        },
    a, b, c);
}

Result AddImplNativeCpu::kernelCF32() {
    return AutomaticIterator<CF32, CF32, CF32>(
        [](const auto& a, const auto& b, auto& c) {
            c = a + b;
        },
    a, b, c);
}

JST_REGISTER_MODULE(AddImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
