#include <cstring>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct UnpadImplNativeCpu : public UnpadImpl,
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

Result UnpadImplNativeCpu::create() {
    JST_CHECK(UnpadImpl::create());

    if (input.dtype() == DataType::F32 &&
        outputUnpadded.dtype() == DataType::F32 &&
        outputPad.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::CF32 &&
        outputUnpadded.dtype() == DataType::CF32 &&
        outputPad.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_UNPAD_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result UnpadImplNativeCpu::computeSubmit() {
    return kernel();
}

Result UnpadImplNativeCpu::kernelF32() {
    const F32* inputData = input.data<F32>();
    F32* unpadData = outputUnpadded.data<F32>();
    F32* padData = outputPad.data<F32>();

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

    // Copy data to unpadded and pad outputs.
    for (U64 outer = 0; outer < outerSize; ++outer) {
        const F32* srcSlice = inputData + outer * inputAxisSize * innerSize;

        // Copy unpadded portion.
        F32* dstUnpad = unpadData + outer * unpadAxisSize * innerSize;
        std::memcpy(dstUnpad, srcSlice, unpadAxisSize * innerSize * sizeof(F32));

        // Copy pad portion.
        F32* dstPad = padData + outer * size * innerSize;
        const F32* srcPad = srcSlice + unpadAxisSize * innerSize;
        std::memcpy(dstPad, srcPad, size * innerSize * sizeof(F32));
    }

    return Result::SUCCESS;
}

Result UnpadImplNativeCpu::kernelCF32() {
    const CF32* inputData = input.data<CF32>();
    CF32* unpadData = outputUnpadded.data<CF32>();
    CF32* padData = outputPad.data<CF32>();

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

    // Copy data to unpadded and pad outputs.
    for (U64 outer = 0; outer < outerSize; ++outer) {
        const CF32* srcSlice = inputData + outer * inputAxisSize * innerSize;

        // Copy unpadded portion.
        CF32* dstUnpad = unpadData + outer * unpadAxisSize * innerSize;
        std::memcpy(dstUnpad, srcSlice, unpadAxisSize * innerSize * sizeof(CF32));

        // Copy pad portion.
        CF32* dstPad = padData + outer * size * innerSize;
        const CF32* srcPad = srcSlice + unpadAxisSize * innerSize;
        std::memcpy(dstPad, srcPad, size * innerSize * sizeof(CF32));
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(UnpadImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
