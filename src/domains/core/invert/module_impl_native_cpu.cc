#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct InvertImplNativeCpu : public InvertImpl,
                             public Runtime::Context,
                             public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();

    std::function<Result()> kernel;
};

Result InvertImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(InvertImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_INVERT_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result InvertImplNativeCpu::computeSubmit() {
    return kernel();
}

Result InvertImplNativeCpu::kernelCF32() {
    const CF32* in = input.data<CF32>();
    CF32* out = output.data<CF32>();
    const U64 size = input.size();

    for (U64 i = 0; i < size; i += 2) {
        out[i] = in[i];
        if (i + 1 < size) {
            out[i + 1] = -in[i + 1];
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(InvertImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
