#include <cmath>
#include <complex>
#include <limits>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FmImplNativeCpu : public FmImpl,
                         public NativeCpuRuntimeContext,
                         public Scheduler::Context {
 public:
    Result validate() override;
    Result computeSubmit() override;
};

Result FmImplNativeCpu::validate() {
    JST_CHECK(FmImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_FM_NATIVE_CPU] Input must be complex (CF32).");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FmImplNativeCpu::computeSubmit() {
    const CF32* inputData = input.data<CF32>();
    F32* outputData = output.data<F32>();
    const F32 refCoeff = ref;
    const Index sampleAxis = *signalAxes.sample;
    const U64 sampleCount = input.shape(sampleAxis);
    const U64 inputSampleStride = input.stride(sampleAxis);
    const U64 outputSampleStride = output.stride(sampleAxis);
    const U64 batchCount = signalAxes.batch
        ? input.shape(*signalAxes.batch) : 1;
    const U64 inputBatchStride = signalAxes.batch
        ? input.stride(*signalAxes.batch) : 0;
    const U64 outputBatchStride = signalAxes.batch
        ? output.stride(*signalAxes.batch) : 0;
    const U64 outputChannelStride = wideBand ? output.stride(input.rank()) : 0;

    for (U64 lane = 0; lane < laneCount; ++lane) {
        U64 coordinates = lane;
        U64 inputLaneOffset = 0;
        U64 outputLaneOffset = 0;
        for (Index axis = input.rank(); axis-- > 0;) {
            if (axis == sampleAxis ||
                (signalAxes.batch && axis == *signalAxes.batch)) {
                continue;
            }
            const U64 coordinate = coordinates % input.shape(axis);
            coordinates /= input.shape(axis);
            inputLaneOffset += coordinate * input.stride(axis);
            outputLaneOffset += coordinate * output.stride(axis);
        }

        CF32 previous = previousSample[lane];
        bool hasPrevious = hasPreviousSample[lane] != 0;
        for (U64 batch = 0; batch < batchCount; ++batch) {
            const U64 inputBatchOffset = inputLaneOffset +
                                         batch * inputBatchStride;
            const U64 outputBatchOffset = outputLaneOffset +
                                          batch * outputBatchStride;
            for (U64 sample = 0; sample < sampleCount; ++sample) {
                const U64 inputOffset = inputBatchOffset +
                                        sample * inputSampleStride;
                const U64 outputOffset = outputBatchOffset +
                                         sample * outputSampleStride;
                const CF32 current = inputData[inputOffset];
                const bool finiteCurrent = std::isfinite(current.real()) &&
                                           std::isfinite(current.imag());
                const bool finitePrevious = std::isfinite(previous.real()) &&
                                            std::isfinite(previous.imag());
                const F32 demodulated = !hasPrevious ? 0.0f :
                    (finiteCurrent && finitePrevious
                        ? std::arg(std::conj(previous) * current) * refCoeff
                        : std::numeric_limits<F32>::quiet_NaN());

                if (!std::isfinite(demodulated)) {
                    outputData[outputOffset] = demodulated;
                    if (wideBand) {
                        outputData[outputOffset + outputChannelStride] =
                            demodulated;
                        auto& state = stereoState[lane];
                        state.pilotPhase += pilotPhaseIncrement;
                        if (state.pilotPhase >= 2.0f * JST_PI) {
                            state.pilotPhase -= 2.0f * JST_PI;
                        }
                    }
                    previous = current;
                    hasPrevious = true;
                    continue;
                }

                if (!wideBand) {
                    if (!deemphasisEnabled) {
                        outputData[outputOffset] = demodulated;
                    } else {
                        auto& deemphasized = narrowDeemphasisState[lane];
                        deemphasized += deemphasisAlpha *
                            (demodulated - deemphasized);
                        outputData[outputOffset] = deemphasized;
                    }
                } else {
                    auto& state = stereoState[lane];
                    const F32 pilotCosine = std::cos(state.pilotPhase);
                    const F32 pilotSine = std::sin(state.pilotPhase);
                    state.pilotCosStage += pilotAlpha *
                        (demodulated * pilotCosine - state.pilotCosStage);
                    state.pilotSinStage += pilotAlpha *
                        (demodulated * pilotSine - state.pilotSinStage);
                    state.pilotCos += pilotAlpha *
                        (state.pilotCosStage - state.pilotCos);
                    state.pilotSin += pilotAlpha *
                        (state.pilotSinStage - state.pilotSin);

                    const F32 sum = applyAudioLowPass(applyBiquad(
                        demodulated, pilotNotch, state.sumNotch),
                        state.sumFilter);
                    const F32 pilotOffset = std::atan2(state.pilotCos,
                                                       state.pilotSin);
                    const F32 differenceCarrier = std::sin(
                        2.0f * (state.pilotPhase + pilotOffset));
                    const F32 difference = applyAudioLowPass(applyBiquad(
                        2.0f * demodulated * differenceCarrier,
                        pilotNotch, state.differenceNotch),
                        state.differenceFilter);

                    F32 left = sum + difference;
                    F32 right = sum - difference;
                    if (deemphasisEnabled) {
                        state.leftDeemphasis += deemphasisAlpha *
                            (left - state.leftDeemphasis);
                        state.rightDeemphasis += deemphasisAlpha *
                            (right - state.rightDeemphasis);
                        left = state.leftDeemphasis;
                        right = state.rightDeemphasis;
                    }

                    outputData[outputOffset] = left;
                    outputData[outputOffset + outputChannelStride] = right;

                    state.pilotPhase += pilotPhaseIncrement;
                    if (state.pilotPhase >= 2.0f * JST_PI) {
                        state.pilotPhase -= 2.0f * JST_PI;
                    }
                }
                previous = current;
                hasPrevious = true;
            }
        }

        previousSample[lane] = previous;
        hasPreviousSample[lane] = U8{1};
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FmImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
