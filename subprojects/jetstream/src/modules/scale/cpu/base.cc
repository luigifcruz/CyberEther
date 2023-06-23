#include "../generic.cc"

namespace Jetstream {

template<typename T>
static inline T scale(const T x, const T min, const T max) {
    return (x - min) / (max - min);
}

template<Device D, typename T>
Result Scale<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Scale compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Scale<D, T>::compute(const RuntimeMetadata&) {
    auto [min, max] = config.range;

    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = scale<T>(input.buffer[i], min, max);
    }

    return Result::SUCCESS;
}

// TODO: Remove in favor of module manifest.
template class Scale<Device::CPU, F32>;
    
}  // namespace Jetstream
