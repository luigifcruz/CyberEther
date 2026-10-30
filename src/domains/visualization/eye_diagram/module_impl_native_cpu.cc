#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include <algorithm>
#include <cmath>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct EyeDiagramImplNativeCpu : public EyeDiagramImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result validate() final;

    Result computeSubmit() override;
};

Result EyeDiagramImplNativeCpu::validate() {
    JST_CHECK(EyeDiagramImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32 &&
        inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_EYE_DIAGRAM_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result EyeDiagramImplNativeCpu::computeSubmit() {
    const U64 windowIntervals = 2 * samplesPerSymbol;
    const F32 width = rangeMax - rangeMin;

    for (U64 i = 0; i < input.size(); ++i) {
        const F32 sample = input.dtype() == DataType::F32 ? input.data<F32>()[i] :
            (component == "real" ? input.data<CF32>()[i].real() :
                                   input.data<CF32>()[i].imag());

        if (!std::isfinite(sample)) {
            initialized = false;
            interval = 0;
            continue;
        }

        if (!initialized) {
            previous = sample;
            initialized = true;
            continue;
        }

        for (U64 j = 0; j < interpolation; ++j) {
            pending[interval * interpolation + j] =
                std::lerp(previous, sample, static_cast<F32>(j) / interpolation);
        }
        previous = sample;

        if (++interval == windowIntervals) {
            pending.back() = sample;

            F32* destination = output.data<F32>();
            for (U64 j = 0; j < pending.size(); ++j) {
                destination[j] =
                    (std::clamp(pending[j], rangeMin, rangeMax) - rangeMin) / width;
            }

            interval = 0;
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(EyeDiagramImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
