#include "../generic.cc"

// Looks like Windows static build crashes if multitheading is enabled.
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct FFT<D, IT, OT>::Impl {
    pocketfft::shape_t shape;
    pocketfft::stride_t i_stride;
    pocketfft::stride_t o_stride;
    pocketfft::shape_t axes;
};

template<Device D, typename IT, typename OT>
FFT<D, IT, OT>::FFT() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
FFT<D, IT, OT>::~FFT() {
    pimpl.reset();
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::createCompute(const Context&) {
    JST_TRACE("Create FFT compute core using CPU backend.");

    for (U64 i = 0; i < input.buffer.rank(); ++i) {
        pimpl->shape.push_back(static_cast<U32>(input.buffer.shape()[i]));
        pimpl->i_stride.push_back(static_cast<U32>(input.buffer.stride()[i]) * sizeof(IT));
        pimpl->o_stride.push_back(static_cast<U32>(input.buffer.stride()[i]) * sizeof(OT));
    }

    // Use the specified axis or default to the last axis
    U64 fft_axis = (config.axis < 0) ? (output.buffer.rank() - 1) : config.axis;
    pimpl->axes.push_back(static_cast<U32>(fft_axis));

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::destroyCompute(const Context&) {
    JST_TRACE("Destroy FFT compute core using CPU backend.");

    pimpl->shape.clear();
    pimpl->i_stride.clear();
    pimpl->o_stride.clear();
    pimpl->axes.clear();

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32, CF32>::compute(const Context&) {
    pocketfft::c2c(pimpl->shape, 
                   pimpl->i_stride, 
                   pimpl->o_stride, 
                   pimpl->axes, 
                   config.forward, 
                   input.buffer.data(), 
                   output.buffer.data(), 
                   1.0f);

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, F32, CF32>::compute(const Context&) {
    pocketfft::r2c(pimpl->shape, 
                   pimpl->i_stride, 
                   pimpl->o_stride, 
                   pimpl->axes, 
                   config.forward, 
                   input.buffer.data(), 
                   output.buffer.data(), 
                   1.0f);

    return Result::SUCCESS;
}

JST_FFT_CPU(JST_INSTANTIATION)
JST_FFT_CPU(JST_BENCHMARK)

}  // namespace Jetstream
