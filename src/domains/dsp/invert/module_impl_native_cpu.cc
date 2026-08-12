#include <cmath>

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
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    template<typename T>
    Result kernelTyped();

    std::function<Result()> kernel;
    U64 axisInnerSize = 1;
    U64 axisLength = 1;
};

Result InvertImplNativeCpu::validate() {
    JST_CHECK(InvertImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32 &&
        inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_INVERT_NATIVE_CPU] Unsupported data type '{}'.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result InvertImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(InvertImpl::create());

    axisInnerSize = 1;
    for (Index axisIndex = resolvedAxis + 1; axisIndex < input.rank(); ++axisIndex) {
        axisInnerSize *= input.shape(axisIndex);
    }
    axisLength = input.shape(resolvedAxis);

    // Register compute kernel.

    if (input.dtype() == DataType::F32) {
        kernel = [this]() { return kernelTyped<F32>(); };
    } else {
        kernel = [this]() { return kernelTyped<CF32>(); };
    }
    return Result::SUCCESS;
}

Result InvertImplNativeCpu::computeSubmit() {
    return kernel();
}

template<typename T>
Result InvertImplNativeCpu::kernelTyped() {
    U64 index = 0;
    const U64 innerSize = axisInnerSize;
    const U64 length = axisLength;

    return AutomaticIterator<const T, CF32>(
        [&index, innerSize, length](const auto& in, auto& out) {
            const U64 axisCoordinate = (index / innerSize) % length;
            const CF32 value(in);
            if ((length & 1ULL) == 0) {
                out = (axisCoordinate & 1ULL) != 0 ? -value : value;
            } else {
                // Odd lengths need an integer-bin phasor instead of (-1)^n.
                const F64 phase = 2.0 * JST_PI *
                                  static_cast<F64>(length / 2) *
                                  static_cast<F64>(axisCoordinate) /
                                  static_cast<F64>(length);
                out = value * CF32(static_cast<F32>(std::cos(phase)),
                                   static_cast<F32>(std::sin(phase)));
            }
            ++index;
        },
        input,
        output);
}

JST_REGISTER_MODULE(InvertImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
