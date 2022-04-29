#include "jetstream/modules/fft/base.hh"

namespace Jetstream {

template<>
FFT<Device::CPU>::FFT(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing FFT module with CPU backend.");

    U64 fftSize = input.buffer.size();

    // Generate FFT window.
    for (U64 i = 0; i < fftSize; i++) {
        float tap;

        tap = 0.5 * (1 - cos(2 * M_PI * i / fftSize));
        tap = (i % 2) == 0 ? tap : -tap;

        CPU.fftWindow.push_back(CF32(tap, 0.0));
    }

    // Generate FFT plan.
    auto inBuf = reinterpret_cast<fftwf_complex*>(input.buffer.data());
    auto outBuf = reinterpret_cast<fftwf_complex*>(output.buffer.data());
    auto direction = (config.direction == Direction::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;
    CPU.fftPlan = fftwf_plan_dft_1d(fftSize, inBuf, outBuf, direction, FFTW_MEASURE);

    JST_INFO("===== FFT Module Configuration");
    JST_INFO("FFT Direction: {}", static_cast<I64>(config.direction));
    JST_INFO("FFT Amplitude (min, max): ({}, {})", config.amplitude.min, config.amplitude.max);
}

template<>
const Result FFT<Device::CPU>::compute() {
    std::cout << "FFT CPU Compute" << std::endl;
    return Result::SUCCESS;
}
    
}  // namespace Jetstream
