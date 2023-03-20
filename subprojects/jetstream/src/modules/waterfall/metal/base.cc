#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Waterfall<D, T>::underlyingCompute(const RuntimeMetadata& meta) {
    auto& runtime = meta.metal;

    auto blitEncoder = runtime.commandBuffer->blitCommandEncoder();
    blitEncoder->copyFromBuffer(input.buffer, NS::UInteger(0),
                                frequencyBins, inc * input.buffer.size_bytes(),
                                NS::UInteger(input.buffer.size_bytes()));
    blitEncoder->endEncoding();

    return Result::SUCCESS;
}

template class Waterfall<Device::Metal, F32>;
    
}  // namespace Jetstream
