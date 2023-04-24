#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Waterfall<D, T>::underlyingCompute(const RuntimeMetadata& meta) {
    auto& runtime = meta.metal;

    auto blitEncoder = runtime.commandBuffer->blitCommandEncoder();

    auto batchByteSize = input.buffer.size_bytes();
    const auto sampleByteSize = batchByteSize / input.buffer.shape(0);
    const auto offset = inc * sampleByteSize;
    const auto size = JST_MIN(batchByteSize, (config.height - inc) * sampleByteSize);

    blitEncoder->copyFromBuffer(input.buffer, 0, frequencyBins, offset, size);
    if (size < batchByteSize) {
        blitEncoder->copyFromBuffer(input.buffer, size, 
            frequencyBins, 0, batchByteSize - size);
    }

    blitEncoder->endEncoding();

    return Result::SUCCESS;
}

template class Waterfall<Device::Metal, F32>;
    
}  // namespace Jetstream
