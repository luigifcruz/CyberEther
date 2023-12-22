#include "../generic.cc"

namespace Jetstream {

template<>
Result FFT<Device::CPU, CF32>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create FFT compute core using CPU backend.");

    auto inBuf = reinterpret_cast<fftwf_complex*>(input.buffer.data());
    auto outBuf = reinterpret_cast<fftwf_complex*>(output.buffer.data());

    I32 M = numberOfOperations;
    I32 N = numberOfElements;
    I32 S = elementStride;
    auto direction = (config.forward) ? FFTW_FORWARD : FFTW_BACKWARD;

    int rank     = 1;      // Number of dimensions
    int n[]      = { N };  // Size of each dimension
    int howmany  = M;      // Number of FFTs
    int idist    = N;      // Distance between consecutive elements in input array
    int odist    = N;      // Distance between consecutive elements in output array
    int istride  = S;      // Stride between successive elements in same FFT
    int ostride  = S;      // Stride between successive elements in same FFT
    int *inembed = n;      // Pointer to array of dimensions for input
    int *onembed = n;      // Pointer to array of dimensions for output

    cpu.fftPlanCF32 = fftwf_plan_many_dft(rank, n, howmany,
                                          inBuf, inembed, istride, idist,
                                          outBuf, onembed, ostride, odist,
                                          direction, FFTW_ESTIMATE);

    if (!cpu.fftPlanCF32) {
        JST_ERROR("Failed to create FFT plan.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32>::destroyCompute(const RuntimeMetadata&) {
    JST_TRACE("Destroy FFT compute core using CPU backend. {}", fmt::ptr(cpu.fftPlanCF32));

    fftwf_destroy_plan(cpu.fftPlanCF32);

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32>::compute(const RuntimeMetadata&) {
    fftwf_execute(cpu.fftPlanCF32);

    return Result::SUCCESS;
}

JST_FFT_CPU(JST_INSTANTIATION);

}  // namespace Jetstream
