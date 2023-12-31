#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
Result Invert<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Invert compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Invert<D, T>::compute(const RuntimeMetadata&) {
    const auto* in = reinterpret_cast<std::pair<T, T>*>(input.buffer.data());
    auto* out = reinterpret_cast<std::pair<T, T>*>(output.buffer.data());

    for (U64 i = 0; i < input.buffer.size() / 2; i++) {
        const auto& [in_even, in_odd] = in[i];
        auto& [out_even, out_odd] = out[i];

        out_even =  in_even;
        out_odd  = -in_odd;
    }

    return Result::SUCCESS;
}

JST_INVERT_CPU(JST_INSTANTIATION)
JST_INVERT_CPU(JST_BENCHMARK)
    
}  // namespace Jetstream
