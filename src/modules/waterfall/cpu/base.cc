#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct Waterfall<D, T>::Impl {};

template<Device D, typename T>
Waterfall<D, T>::Waterfall() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Waterfall<D, T>::~Waterfall() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Waterfall<D, T>::GImpl::underlyingCompute(Waterfall<D, T>& m, const Context&) {
    const auto totalSize = m.input.buffer.size();
    const auto fftSize = numberOfElements;
    const auto offset = inc * fftSize;
    const auto size = JST_MIN(totalSize, (m.config.height - inc) * fftSize);

    std::copy(m.input.buffer.begin(), m.input.buffer.begin() + size, frequencyBins.data() + offset);
    if (size < totalSize) {
        std::copy(m.input.buffer.begin() + size, m.input.buffer.end(), frequencyBins.data());
    }

    return Result::SUCCESS;
}

JST_WATERFALL_CPU(JST_INSTANTIATION)
JST_WATERFALL_CPU(JST_BENCHMARK)

}  // namespace Jetstream
