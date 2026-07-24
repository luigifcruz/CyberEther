#include <limits>

#include <jetstream/memory/macros.hh>
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
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelF32();
    Result kernelCF32();

    std::function<Result()> kernel;
};

Result MultiplyImplNativeCpu::validate() {
    JST_CHECK(MultiplyImpl::validate());

    if (!inputs().contains("a") || !inputs().contains("b")) {
        return Result::SUCCESS;
    }

    const Tensor& tensorA = inputs().at("a").tensor;
    const Tensor& tensorB = inputs().at("b").tensor;
    if (!tensorA.validShape() || !tensorB.validShape() ||
        tensorA.size() == 0 || tensorB.size() == 0) {
        return Result::SUCCESS;
    }

    if (tensorA.dtype() != tensorB.dtype()) {
        JST_ERROR("[MODULE_MULTIPLY_NATIVE_CPU] Input data types '{}' and '{}' do not match.",
                  tensorA.dtype(), tensorB.dtype());
        return Result::ERROR;
    }

    if (tensorA.dtype() != DataType::F32 && tensorA.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_MULTIPLY_NATIVE_CPU] Unsupported data type '{}'.",
                  tensorA.dtype());
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_MULTIPLY_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result MultiplyImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(MultiplyImpl::create());

    // Register compute kernel.

    if (a.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
    } else {
        kernel = [this]() { return kernelCF32(); };
    }

    return Result::SUCCESS;
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
