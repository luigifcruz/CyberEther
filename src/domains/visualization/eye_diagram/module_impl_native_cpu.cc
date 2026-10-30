#include <algorithm>
#include <cmath>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct EyeDiagramImplNativeCpu : public EyeDiagramImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
    Result computeSubmit() override;
};

Result EyeDiagramImplNativeCpu::computeSubmit() {
    const U64 windowIntervals = 2 * samplesPerSymbol;
    const F32 width = rangeMax - rangeMin;
    for (U64 i = 0; i < input.size(); ++i) {
        const F32 sample = input.dtype() == DataType::F32 ? input.data<F32>()[i] :
            (component == "real" ? input.data<CF32>()[i].real() :
                                   input.data<CF32>()[i].imag());
        if (!std::isfinite(sample)) {
            // Do not interpolate through a missing/non-finite sample.
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
                // Clip before scaling to avoid overflow for extreme finite inputs.
                destination[j] =
                    (std::clamp(pending[j], rangeMin, rangeMax) - rangeMin) / width;
            }
            interval = 0;
        }
    }
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(EyeDiagramImplNativeCpu, DeviceType::CPU,
                    RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
