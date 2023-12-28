#include "../generic.cc"

namespace Jetstream {

template<>
Result FFT<Device::CPU, CF32>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create FFT compute core using CPU backend.");

    for (U64 i = 0; i < input.buffer.rank(); ++i) {
        cpu.shape.push_back(static_cast<U32>(input.buffer.shape()[i]));
        cpu.stride.push_back(static_cast<U32>(input.buffer.stride()[i]) * sizeof(std::complex<F32>));
    }

    cpu.axes.push_back(output.buffer.rank() - 1);

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32>::destroyCompute(const RuntimeMetadata&) {
    JST_TRACE("Destroy FFT compute core using CPU backend.");

    cpu.shape.clear();
    cpu.stride.clear();
    cpu.axes.clear();

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32>::compute(const RuntimeMetadata&) {
    pocketfft::c2c(cpu.shape, 
                   cpu.stride, 
                   cpu.stride, 
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
