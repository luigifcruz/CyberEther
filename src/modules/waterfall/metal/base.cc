#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Waterfall<D, T>::underlyingCompute(const RuntimeMetadata& meta) {
    auto& runtime = meta.metal;

    auto blitEncoder = runtime.commandBuffer->blitCommandEncoder();

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

JST_WATERFALL_METAL(JST_INSTANTIATION);
    
}  // namespace Jetstream
