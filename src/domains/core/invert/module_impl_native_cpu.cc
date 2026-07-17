#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct InvertImplNativeCpu : public InvertImpl,
                             public NativeCpuRuntimeContext,
                             public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();

    std::function<Result()> kernel;
    U64 axisInnerSize = 1;
    U64 axisLength = 1;
};

Result InvertImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(InvertImpl::create());

    axisInnerSize = 1;
    for (Index axisIndex = resolvedAxis + 1; axisIndex < input.rank(); ++axisIndex) {
        axisInnerSize *= input.shape(axisIndex);
    }
    axisLength = input.shape(resolvedAxis);

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
    U64 index = 0;
    const U64 innerSize = axisInnerSize;
    const U64 length = axisLength;

    return AutomaticIterator<const CF32, CF32>(
        [&index, innerSize, length](const auto& in, auto& out) {
            const U64 axisCoordinate = (index / innerSize) % length;
            out = (axisCoordinate & 1ULL) != 0 ? -in : in;
            ++index;
        },
        input,
        output);
}

JST_REGISTER_MODULE(InvertImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
