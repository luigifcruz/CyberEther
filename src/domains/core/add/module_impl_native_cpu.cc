#include <limits>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/memory/macros.hh>
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
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelF32();
    Result kernelCF32();

    std::function<Result()> kernel;
};

Result AddImplNativeCpu::validate() {
    JST_CHECK(AddImpl::validate());

    if (!inputs().contains("a") || !inputs().contains("b")) {
        return Result::SUCCESS;
    }

    const auto dtypeA = inputs().at("a").tensor.dtype();
    const auto dtypeB = inputs().at("b").tensor.dtype();
    if (dtypeA != dtypeB) {
        JST_ERROR("[MODULE_ADD_NATIVE_CPU] Input data types '{}' and '{}' do not match.",
                  dtypeA, dtypeB);
        return Result::ERROR;
    }

    if (dtypeA != DataType::F32 && dtypeA != DataType::CF32) {
        JST_ERROR("[MODULE_ADD_NATIVE_CPU] Unsupported data type '{}'.", dtypeA);
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_ADD_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AddImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(AddImpl::create());

    // Register compute kernel.

    if (a.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
    } else {
        kernel = [this]() { return kernelCF32(); };
    }

    return Result::SUCCESS;
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
