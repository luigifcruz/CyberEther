#include <cmath>
#include <functional>
#include <limits>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ComparatorImplNativeCpu : public ComparatorImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelF32();
    Result kernelF64();
    Result kernelCF32();
    Result kernelCF64();

    template<typename T, typename E>
    Result comparePair(Tensor& reference, Tensor& candidate, bool firstPair);

    void publishStats(F64 maxDiff, F64 sumAbs, F64 sumSq, U64 count);
    void accumulateDiff(F64 diff, F64& maxDiff, F64& sumAbs, F64& sumSq);

    std::function<Result()> kernel;
};

Result ComparatorImplNativeCpu::create() {
    JST_CHECK(ComparatorImpl::create());

    const auto dtype = inputTensors.front().dtype();

    if (dtype == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    if (dtype == DataType::F64) {
        kernel = [this]() { return kernelF64(); };
        return Result::SUCCESS;
    }

    if (dtype == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    if (dtype == DataType::CF64) {
        kernel = [this]() { return kernelCF64(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_COMPARATOR_NATIVE_CPU] Unsupported data type '{}'.", dtype);
    return Result::ERROR;
}

Result ComparatorImplNativeCpu::computeSubmit() {
    return kernel();
}

void ComparatorImplNativeCpu::publishStats(const F64 maxDiff,
                                           const F64 sumAbs,
                                           const F64 sumSq,
                                           const U64 count) {
    const F64 meanDiff = count > 0 ? (sumAbs / static_cast<F64>(count)) : 0.0;
    const F64 mse = count > 0 ? (sumSq / static_cast<F64>(count)) : 0.0;
    const bool match = std::isfinite(maxDiff) && maxDiff <= tolerance;

    maxDiffState.publish(maxDiff);
    meanDiffState.publish(meanDiff);
    mseState.publish(mse);
    matchState.publish(match);
}

template<typename T, typename E>
Result ComparatorImplNativeCpu::comparePair(Tensor& reference,
                                            Tensor& candidate,
                                            const bool firstPair) {
    return AutomaticIterator<T, T, E>(
        [&](const auto& a, const auto& b, auto& e) {
            const E d = static_cast<E>(std::abs(a - b));
            if (firstPair || !std::isfinite(static_cast<F64>(d)) ||
                (std::isfinite(static_cast<F64>(e)) && d > e)) {
                e = d;
            }
        },
        reference,
        candidate,
        error);
}

void ComparatorImplNativeCpu::accumulateDiff(const F64 diff,
                                             F64& maxDiff,
                                             F64& sumAbs,
                                             F64& sumSq) {
    if (std::isfinite(diff)) {
        maxDiff = std::max(maxDiff, diff);
    } else {
        maxDiff = std::numeric_limits<F64>::infinity();
    }

    sumAbs += diff;
    sumSq += diff * diff;
}

Result ComparatorImplNativeCpu::kernelF32() {
    for (U64 i = 1; i < inputTensors.size(); ++i) {
        JST_CHECK(comparePair<F32, F32>(inputTensors[0], inputTensors[i], i == 1));
    }

    F64 maxDiff = 0.0;
    F64 sumAbs = 0.0;
    F64 sumSq = 0.0;
    const U64 count = error.size();

    JST_CHECK(AutomaticIterator<F32>(
        [&](const auto& e) {
            const F64 d = static_cast<F64>(e);
            accumulateDiff(d, maxDiff, sumAbs, sumSq);
        },
        error));

    publishStats(maxDiff, sumAbs, sumSq, count);

    return Result::SUCCESS;
}

Result ComparatorImplNativeCpu::kernelF64() {
    for (U64 i = 1; i < inputTensors.size(); ++i) {
        JST_CHECK(comparePair<F64, F64>(inputTensors[0], inputTensors[i], i == 1));
    }

    F64 maxDiff = 0.0;
    F64 sumAbs = 0.0;
    F64 sumSq = 0.0;
    const U64 count = error.size();

    JST_CHECK(AutomaticIterator<F64>(
        [&](const auto& e) {
            accumulateDiff(e, maxDiff, sumAbs, sumSq);
        },
        error));

    publishStats(maxDiff, sumAbs, sumSq, count);

    return Result::SUCCESS;
}

Result ComparatorImplNativeCpu::kernelCF32() {
    for (U64 i = 1; i < inputTensors.size(); ++i) {
        JST_CHECK(comparePair<CF32, F32>(inputTensors[0], inputTensors[i], i == 1));
    }

    F64 maxDiff = 0.0;
    F64 sumAbs = 0.0;
    F64 sumSq = 0.0;
    const U64 count = error.size();

    JST_CHECK(AutomaticIterator<F32>(
        [&](const auto& e) {
            const F64 d = static_cast<F64>(e);
            accumulateDiff(d, maxDiff, sumAbs, sumSq);
        },
        error));

    publishStats(maxDiff, sumAbs, sumSq, count);

    return Result::SUCCESS;
}

Result ComparatorImplNativeCpu::kernelCF64() {
    for (U64 i = 1; i < inputTensors.size(); ++i) {
        JST_CHECK(comparePair<CF64, F64>(inputTensors[0], inputTensors[i], i == 1));
    }

    F64 maxDiff = 0.0;
    F64 sumAbs = 0.0;
    F64 sumSq = 0.0;
    const U64 count = error.size();

    JST_CHECK(AutomaticIterator<F64>(
        [&](const auto& e) {
            accumulateDiff(e, maxDiff, sumAbs, sumSq);
        },
        error));

    publishStats(maxDiff, sumAbs, sumSq, count);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(ComparatorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
