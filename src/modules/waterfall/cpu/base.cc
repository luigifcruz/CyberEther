#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Waterfall<D, T>::underlyingCompute(const RuntimeMetadata&) {
    const auto totalSize = input.buffer.size();
    const auto fftSize = numberOfElements;
    const auto offset = inc * fftSize;
    const auto size = JST_MIN(totalSize, (config.height - inc) * fftSize);

    std::copy(input.buffer.begin(), input.buffer.begin() + size, frequencyBins.data() + offset);
    if (size < totalSize) {
        std::copy(input.buffer.begin() + size, input.buffer.end(), frequencyBins.data());
    }

    return Result::SUCCESS;
}

JST_WATERFALL_CPU(JST_INSTANTIATION)
JST_WATERFALL_CPU(JST_BENCHMARK)
    
}  // namespace Jetstream
