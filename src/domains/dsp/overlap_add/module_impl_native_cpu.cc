#include <cstring>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct OverlapAddImplNativeCpu : public OverlapAddImpl,
                                 public Runtime::Context,
                                 public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();
    Result kernelF32();

    std::function<Result()> kernel;

    // Precomputed strides for coordinate conversion.
    std::vector<U64> bufferStrides;
    std::vector<U64> overlapStrides;
    std::vector<U64> prevOverlapStrides;
};

Result OverlapAddImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(OverlapAddImpl::create());

    if (inputBuffer.dtype() != inputOverlap.dtype()) {
        JST_ERROR("[MODULE_OVERLAP_ADD_NATIVE_CPU] Input dtype mismatch: "
                  "buffer is {}, overlap is {}.",
                  inputBuffer.dtype(),
                  inputOverlap.dtype());
        return Result::ERROR;
    }

    // Precompute row-major strides.

    const U64 rank = inputBuffer.rank();

    bufferStrides.resize(rank);
    overlapStrides.resize(rank);
    prevOverlapStrides.resize(rank);

    bufferStrides[rank - 1] = 1;
    overlapStrides[rank - 1] = 1;
    prevOverlapStrides[rank - 1] = 1;

    for (U64 d = rank - 1; d > 0; --d) {
        bufferStrides[d - 1] = bufferStrides[d] * inputBuffer.shape(d);
        overlapStrides[d - 1] = overlapStrides[d] * inputOverlap.shape(d);
        prevOverlapStrides[d - 1] = prevOverlapStrides[d] * previousOverlap.shape(d);
    }

    // Register compute kernel.

    if (inputBuffer.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    if (inputBuffer.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_OVERLAP_ADD_NATIVE_CPU] Unsupported input "
              "data type: {}.",
              inputBuffer.dtype());
    return Result::ERROR;
}

Result OverlapAddImplNativeCpu::computeSubmit() {
    return kernel();
}

template<typename T>
static Result overlapAddKernel(const Tensor& inputBuffer,
                               const Tensor& inputOverlap,
                               Tensor& output,
                               Tensor& previousOverlap,
                               const std::vector<U64>& bufStrides,
                               const std::vector<U64>& ovlStrides,
                               const std::vector<U64>& prevOvlStrides) {
    const U64 rank = inputBuffer.rank();
    const U64 totalBuf = inputBuffer.size();
    const U64 totalOvl = inputOverlap.size();
    const U64 totalPrev = previousOverlap.size();

    const T* bufPtr = inputBuffer.data<T>();
    const T* ovlPtr = inputOverlap.data<T>();
    T* outPtr = output.data<T>();
    T* prevPtr = previousOverlap.data<T>();

    // 1. Copy input buffer to output.
    std::memcpy(outPtr, bufPtr, totalBuf * sizeof(T));

    // 2. Add overlap to output buffer.
    std::vector<U64> coords(rank);

    for (U64 i = 0; i < totalOvl; ++i) {
        // Convert linear overlap index to coordinates.
        U64 rem = i;
        for (U64 d = 0; d < rank; ++d) {
            coords[d] = rem / ovlStrides[d];
            rem %= ovlStrides[d];
        }

        // Compute output linear index from overlap coords.
        U64 outIdx = 0;
        for (U64 d = 0; d < rank; ++d) {
            outIdx += coords[d] * bufStrides[d];
        }

        if (rank == 1 || coords[0] == 0) {
            // First batch: add stored previous overlap.
            U64 prevIdx = 0;
            for (U64 d = 0; d < rank; ++d) {
                prevIdx += coords[d] * prevOvlStrides[d];
            }
            outPtr[outIdx] += prevPtr[prevIdx];
        } else {
            // Other batches: add overlap from previous batch.
            coords[0] -= 1;
            U64 srcIdx = 0;
            for (U64 d = 0; d < rank; ++d) {
                srcIdx += coords[d] * ovlStrides[d];
            }
            outPtr[outIdx] += ovlPtr[srcIdx];
        }
    }

    // 3. Store last batch of overlap for next invocation.
    if (rank == 1) {
        std::memcpy(prevPtr, ovlPtr, totalPrev * sizeof(T));
    } else {
        const U64 lastBatch = inputOverlap.shape(0) - 1;

        for (U64 i = 0; i < totalPrev; ++i) {
            // Convert linear previousOverlap index to coords.
            U64 rem = i;
            for (U64 d = 0; d < rank; ++d) {
                coords[d] = rem / prevOvlStrides[d];
                rem %= prevOvlStrides[d];
            }

            // Read from last batch of overlap.
            coords[0] = lastBatch;
            U64 srcIdx = 0;
            for (U64 d = 0; d < rank; ++d) {
                srcIdx += coords[d] * ovlStrides[d];
            }
            prevPtr[i] = ovlPtr[srcIdx];
        }
    }

    return Result::SUCCESS;
}

Result OverlapAddImplNativeCpu::kernelCF32() {
    return overlapAddKernel<CF32>(inputBuffer,
                                  inputOverlap,
                                  output,
                                  previousOverlap,
                                  bufferStrides,
                                  overlapStrides,
                                  prevOverlapStrides);
}

Result OverlapAddImplNativeCpu::kernelF32() {
    return overlapAddKernel<F32>(inputBuffer,
                                 inputOverlap,
                                 output,
                                 previousOverlap,
                                 bufferStrides,
                                 overlapStrides,
                                 prevOverlapStrides);
}

JST_REGISTER_MODULE(OverlapAddImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
