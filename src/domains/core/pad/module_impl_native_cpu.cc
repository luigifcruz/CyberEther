#include <cstring>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct PadImplNativeCpu : public PadImpl,
                          public Runtime::Context,
                          public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelF32();
    Result kernelCF32();

    std::function<Result()> kernel;
};

Result PadImplNativeCpu::create() {
    JST_CHECK(PadImpl::create());

    if (input.dtype() == DataType::F32 && output.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::CF32 && output.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_PAD_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result PadImplNativeCpu::computeSubmit() {
    return kernel();
}

Result PadImplNativeCpu::kernelF32() {
    const F32* inputData = input.data<F32>();
    F32* outputData = output.data<F32>();

    const Shape& inShape = input.shape();
    const U64 rank = input.rank();

    // Calculate the number of slices before the axis and the size of each slice.
    U64 outerSize = 1;
    for (U64 i = 0; i < axis; ++i) {
        outerSize *= inShape[i];
    }

    U64 innerSize = 1;
    for (U64 i = axis + 1; i < rank; ++i) {
        innerSize *= inShape[i];
    }

    // Copy data and add padding for each outer slice.
    for (U64 outer = 0; outer < outerSize; ++outer) {
        // Copy original data.
        const F32* srcSlice = inputData + outer * inputAxisSize * innerSize;
        F32* dstSlice = outputData + outer * outputAxisSize * innerSize;

        std::memcpy(dstSlice, srcSlice, inputAxisSize * innerSize * sizeof(F32));

        // Zero-fill padding.
        F32* padStart = dstSlice + inputAxisSize * innerSize;
        std::memset(padStart, 0, size * innerSize * sizeof(F32));
    }

    return Result::SUCCESS;
}

Result PadImplNativeCpu::kernelCF32() {
    const CF32* inputData = input.data<CF32>();
    CF32* outputData = output.data<CF32>();

    const Shape& inShape = input.shape();
    const U64 rank = input.rank();

    // Calculate the number of slices before the axis and the size of each slice.
    U64 outerSize = 1;
    for (U64 i = 0; i < axis; ++i) {
        outerSize *= inShape[i];
    }

    U64 innerSize = 1;
    for (U64 i = axis + 1; i < rank; ++i) {
        innerSize *= inShape[i];
    }

    // Copy data and add padding for each outer slice.
    for (U64 outer = 0; outer < outerSize; ++outer) {
        // Copy original data.
        const CF32* srcSlice = inputData + outer * inputAxisSize * innerSize;
        CF32* dstSlice = outputData + outer * outputAxisSize * innerSize;

        std::memcpy(dstSlice, srcSlice, inputAxisSize * innerSize * sizeof(CF32));

        // Zero-fill padding.
        CF32* padStart = dstSlice + inputAxisSize * innerSize;
        const U64 padCount = size * innerSize;
        std::fill(padStart, padStart + padCount, CF32{0.0f, 0.0f});
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PadImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
