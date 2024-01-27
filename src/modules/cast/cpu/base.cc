#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result Cast<D, IT, OT>::createCompute(const Context&) {
    JST_TRACE("Create Cast compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Cast<D, IT, OT>::compute(const Context&) {
    const IT maxValue = std::numeric_limits<OT>::max();
    const IT minValue = std::numeric_limits<OT>::min();

    for (U64 i = 0; i < input.buffer.size(); i++) {
        IT scaledValue = input.buffer[i] * config.scaler;
        IT clampedValue = std::clamp(scaledValue, minValue, maxValue);
        output.buffer[i]  = static_cast<OT>(clampedValue);
    }

    return Result::SUCCESS;
}

JST_CAST_CPU(JST_INSTANTIATION)
JST_CAST_CPU(JST_BENCHMARK)

}  // namespace Jetstream
