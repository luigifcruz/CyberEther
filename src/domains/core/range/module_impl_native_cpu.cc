#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct RangeImplNativeCpu : public RangeImpl,
                            public Runtime::Context,
                            public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelF32();

    std::function<Result()> kernel;
};

Result RangeImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(RangeImpl::create());

    // Register compute kernel.

    if ((input.dtype() == DataType::F32) and
        (output.dtype() == DataType::F32)) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_RANGE_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result RangeImplNativeCpu::computeSubmit() {
    if (!input.data() || !output.data()) {
        return Result::SUCCESS;
    }
    return kernel();
}

Result RangeImplNativeCpu::kernelF32() {
    const F32 scale = scalingCoeff;
    const F32 offset = offsetCoeff;

    return AutomaticIterator<F32, F32>(
        [scale, offset](const auto& in, auto& out) {
            out = in * scale + offset;
        },
    input, output);
}

JST_REGISTER_MODULE(RangeImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
