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
Result Waterfall<D, T>::underlyingCompute(const Context& ctx) {
    auto blitEncoder = ctx.metal->commandBuffer()->blitCommandEncoder();

    auto batchByteSize = input.buffer.size_bytes();
    const auto sampleByteSize = batchByteSize / numberOfBatches;
    const auto offset = inc * sampleByteSize;
    const auto size = JST_MIN(batchByteSize, (config.height - inc) * sampleByteSize);

    blitEncoder->copyFromBuffer(input.buffer.data(), 0, frequencyBins.data(), offset, size);
    if (size < batchByteSize) {
        blitEncoder->copyFromBuffer(input.buffer.data(), size, 
            frequencyBins.data(), 0, batchByteSize - size);
    }

    blitEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_WATERFALL_METAL(JST_INSTANTIATION)
JST_WATERFALL_METAL(JST_BENCHMARK)
    
}  // namespace Jetstream
