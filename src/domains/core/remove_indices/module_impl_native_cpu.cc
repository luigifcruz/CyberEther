#include <cstring>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct RemoveIndicesImplNativeCpu : public RemoveIndicesImpl,
                                    public NativeCpuRuntimeContext,
                                    public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;

 private:
    struct CopyRun {
        U64 inputOffset;
        U64 outputOffset;
        U64 sizeBytes;
    };

    std::vector<CopyRun> copyRuns;
    U64 outerSize = 0;
    U64 inputOuterBytes = 0;
    U64 outputOuterBytes = 0;
};

Result RemoveIndicesImplNativeCpu::create() {
    JST_CHECK(RemoveIndicesImpl::create());

    copyRuns.clear();
    if (input.contiguous()) {
        U64 innerBytes = input.elementSize();
        for (Index axis = resolvedAxis + 1; axis < input.rank(); ++axis) {
            innerBytes *= input.shape(axis);
        }

        inputOuterBytes = input.shape(resolvedAxis) * innerBytes;
        outputOuterBytes = output.shape(resolvedAxis) * innerBytes;
        outerSize = input.sizeBytes() / inputOuterBytes;

        // Adjacent surviving entries can be copied together, including whole
        // antenna ranges when removing from the leading axis of a packet.
        for (U64 begin = 0; begin < keptIndices.size();) {
            U64 end = begin + 1;
            while (end < keptIndices.size() && keptIndices[end] == keptIndices[end - 1] + 1) {
                ++end;
            }
            copyRuns.push_back({keptIndices[begin] * innerBytes,
                                begin * innerBytes,
                                (end - begin) * innerBytes});
            begin = end;
        }
    }

    return Result::SUCCESS;
}

Result RemoveIndicesImplNativeCpu::computeSubmit() {
    const auto* inputData = input.data<U8>();
    auto* outputData = output.data<U8>();
    if (!inputData || !outputData) {
        JST_ERROR("[MODULE_REMOVE_INDICES_NATIVE_CPU] Missing input or output buffer.");
        return Result::ERROR;
    }

    if (input.contiguous()) {
        for (U64 outer = 0; outer < outerSize; ++outer) {
            for (const auto& run : copyRuns) {
                std::memcpy(outputData + outer * outputOuterBytes + run.outputOffset,
                            inputData + outer * inputOuterBytes + run.inputOffset,
                            run.sizeBytes);
            }
        }
        return Result::SUCCESS;
    }

    // Strides are in elements; data() already includes a CPU view's byte offset.
    const U64 elementSize = input.elementSize();
    for (U64 index = 0; index < output.size(); ++index) {
        U64 remaining = index;
        U64 sourceIndex = 0;
        for (Index axis = output.rank(); axis-- > 0;) {
            U64 coordinate = remaining % output.shape(axis);
            remaining /= output.shape(axis);
            if (axis == resolvedAxis) {
                coordinate = keptIndices[coordinate];
            }
            sourceIndex += coordinate * input.stride(axis);
        }
        std::memcpy(outputData + index * elementSize,
                    inputData + sourceIndex * elementSize,
                    elementSize);
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(RemoveIndicesImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
