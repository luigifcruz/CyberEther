#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct Waterfall<D, T>::Impl {};

template<Device D, typename T>
Waterfall<D, T>::Waterfall() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Waterfall<D, T>::~Waterfall() {
    pimpl.reset();
}

template<Device D, typename T>
Result Waterfall<D, T>::underlyingCompute(const Context& ctx) {
    const auto totalSize = input.buffer.size_bytes();
    const auto fftSize = numberOfElements * sizeof(T);
    const auto offset = inc * fftSize;
    const auto size = JST_MIN(totalSize, (config.height - inc) * fftSize);

    const auto direction = (input.buffer.device_native()) ? cudaMemcpyDeviceToDevice : 
                                                            cudaMemcpyHostToDevice;

    uint8_t* bins = reinterpret_cast<uint8_t*>(frequencyBins.data());
    uint8_t* in = reinterpret_cast<uint8_t*>(input.buffer.data());

    JST_CUDA_CHECK(cudaMemcpyAsync(bins + offset, 
                                   in, 
                                   size, 
                                   direction, 
                                   ctx.cuda->stream()), [&]{
        JST_ERROR("Failed to copy data to CUDA device: {}.", err);
    });

    if (size < totalSize) {
        JST_CUDA_CHECK(cudaMemcpyAsync(bins, 
                                       in + size, 
                                       totalSize - size, 
                                       direction, 
                                       ctx.cuda->stream()), [&]{
            JST_ERROR("Failed to copy data to CUDA device: {}.", err);
        });
    }

    return Result::SUCCESS;
}

JST_WATERFALL_CUDA(JST_INSTANTIATION)
JST_WATERFALL_CUDA(JST_BENCHMARK)
    
}  // namespace Jetstream
