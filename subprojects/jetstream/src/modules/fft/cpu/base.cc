#include "../generic.cc"

namespace Jetstream {

template<>
Result FFT<Device::CPU, CF32>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create FFT compute core using CPU backend.");

    auto inBuf = reinterpret_cast<fftwf_complex*>(input.buffer.data());
    auto outBuf = reinterpret_cast<fftwf_complex*>(output.buffer.data());

    const I32 M = input.buffer.shape(0);
    const I32 N = input.buffer.shape(1);
    auto direction = (config.direction == Direction::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;

    int rank     = 1;      // Number of dimensions
    int n[]      = { N };  // Size of each dimension
    int howmany  = M;      // Number of FFTs
    int idist    = N;      // Distance between consecutive elements in input array
    int odist    = N;      // Distance between consecutive elements in output array
    int istride  = 1;      // Stride between successive elements in same FFT
    int ostride  = 1;      // Stride between successive elements in same FFT
    int *inembed = n;      // Pointer to array of dimensions for input
    int *onembed = n;      // Pointer to array of dimensions for output

    cpu.fftPlanCF32 = fftwf_plan_many_dft(rank, n, howmany, 
                                          inBuf, inembed, istride, idist,
                                          outBuf, onembed, ostride, odist,
                                          direction, FFTW_ESTIMATE);

    if (!cpu.fftPlanCF32) {
        JST_FATAL("Failed to create FFT plan.");
        JST_CHECK_THROW(Result::ERROR);   
    }

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32>::destroyCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Destroy FFT compute core using CPU backend.");

    return Result::SUCCESS;
}

template<>
Result FFT<Device::CPU, CF32>::compute(const RuntimeMetadata& meta) {
    fftwf_execute(cpu.fftPlanCF32);

    return Result::SUCCESS;
}

template class FFT<Device::CPU, CF32>;
    
}  // namespace Jetstream
