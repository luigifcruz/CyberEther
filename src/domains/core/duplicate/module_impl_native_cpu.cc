#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct DuplicateImplNativeCpu : public DuplicateImpl,
                                public Runtime::Context,
                                public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();
    Result kernelF32();

    std::function<Result()> kernel;
};

Result DuplicateImplNativeCpu::create() {
    JST_CHECK(DuplicateImpl::create());

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_DUPLICATE_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result DuplicateImplNativeCpu::computeSubmit() {
    return kernel();
}

Result DuplicateImplNativeCpu::kernelCF32() {
    if (input.contiguous() && input.sizeBytes() == input.buffer().sizeBytes()) {
        JST_CHECK(output.copyFrom(input));
    } else {
        JST_CHECK(AutomaticIterator<const CF32, CF32>(
            [](const auto& in, auto& out) {
                out = in;
            },
        input, output));
    }

    return Result::SUCCESS;
}

Result DuplicateImplNativeCpu::kernelF32() {
    if (input.contiguous() && input.sizeBytes() == input.buffer().sizeBytes()) {
        JST_CHECK(output.copyFrom(input));
    } else {
        JST_CHECK(AutomaticIterator<const F32, F32>(
            [](const auto& in, auto& out) {
                out = in;
            },
        input, output));
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DuplicateImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
