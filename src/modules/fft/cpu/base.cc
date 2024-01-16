#include "../generic.cc"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create FFT compute core using CPU backend.");

    for (U64 i = 0; i < input.buffer.rank(); ++i) {
        cpu.shape.push_back(static_cast<U32>(input.buffer.shape()[i]));
        cpu.i_stride.push_back(static_cast<U32>(input.buffer.stride()[i]) * sizeof(IT));
        cpu.o_stride.push_back(static_cast<U32>(input.buffer.stride()[i]) * sizeof(OT));
    }

    cpu.axes.push_back(output.buffer.rank() - 1);

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::destroyCompute(const RuntimeMetadata&) {
    JST_TRACE("Destroy FFT compute core using CPU backend.");

    cpu.shape.clear();
    cpu.i_stride.clear();
    cpu.o_stride.clear();
    cpu.axes.clear();

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32, CF32>::compute(const RuntimeMetadata&) {
    pocketfft::c2c(cpu.shape, 
                   cpu.i_stride, 
                   cpu.o_stride, 
                   cpu.axes, 
                   config.forward, 
                   input.buffer.data(), 
                   output.buffer.data(), 
                   1.0f);

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, F32, CF32>::compute(const RuntimeMetadata&) {
    pocketfft::r2c(cpu.shape, 
                   cpu.i_stride, 
                   cpu.o_stride, 
                   cpu.axes, 
                   config.forward, 
                   input.buffer.data(), 
                   output.buffer.data(), 
                   1.0f);

    return Result::SUCCESS;
}

JST_FFT_CPU(JST_INSTANTIATION)
JST_FFT_CPU(JST_BENCHMARK)

}  // namespace Jetstream
