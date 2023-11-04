#include "jetstream/modules/filter.hh"

namespace Jetstream {

inline F32 sinc(F32 x) {
    return (x == 0) ? 1.0 : sin(M_PI * x) / (M_PI * x);
}

inline F32 n(U64 len, U64 index) {
    return static_cast<F32>(index - static_cast<float>(len - 1) / 2);
}

template<Device D, typename T>
Result Filter<D, T>::generateSincFunction() {
    std::fill(sincCoeffs.begin(), sincCoeffs.end(), 0.0);

    for (U64 b = 0; b < config.shape[0]; b++) {
        for (U64 i = 0; i < config.numberOfTaps; i++) {
            sincCoeffs[{b, i}] = sinc(2.0 * filterWidth * (i - (config.numberOfTaps - 1) / 2.0));
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Filter<D, T>::generateWindow() {
    std::fill(windowCoeffs.begin(), windowCoeffs.end(), 0.0);

    for (U64 b = 0; b < config.shape[0]; b++) {
        for (U64 i = 0; i < config.numberOfTaps; i++) {
            windowCoeffs[{b, i}] = 0.42 - 0.50 * cos(2.0 * M_PI * i / (config.numberOfTaps - 1)) + \
                                   0.08 * cos(4.0 * M_PI * i / (config.numberOfTaps - 1));
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Filter<D, T>::generateUpconvert() {
    std::fill(upconvertCoeffs.begin(), upconvertCoeffs.end(), T(0.0, 0.0));

    std::complex<float> j(0, 1);
    for (U64 b = 0; b < config.shape[0]; b++) {
        for (U64 i = 0; i < config.numberOfTaps; i++) {
            upconvertCoeffs[{b, i}] = std::exp(j * 2.0f * (float)M_PI * n(config.numberOfTaps, i) * filterOffset);
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Filter<D, T>::calculateFractions() {
    config.numberOfTaps = config.numberOfTaps | 1;
    filterWidth = (config.filterSampleRate / config.signalSampleRate) / 2.0;
    filterOffset = (config.filterCenter / (config.signalSampleRate / 2.0)) / 2.0;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Filter<D, T>::bakeFilter() {
    // Merge all coefficients.
    for (U64 b = 0; b < config.shape[0]; b++) {
        for (U64 i = 0; i < config.shape[1]; i++) {
            const auto coeff = (sincCoeffs[{b, i}] * windowCoeffs[{b, i}]) * upconvertCoeffs[{b, i}];
            scratchCoeffs[{b, i}] = ((i % 2) == 0 and config.linearFrequency) ? coeff : -coeff;
        }
    }

    // Convert filter to the frequency domain.
    fftwf_execute(plan);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Filter<D, T>::create() {
    JST_DEBUG("Initializing Filter module.");

    // Allocate internal data.
    sincCoeffs = Tensor<D, typename T::value_type>(config.shape);
    windowCoeffs = Tensor<D, typename T::value_type>(config.shape);
    upconvertCoeffs = Tensor<D, T>(config.shape);
    scratchCoeffs = Tensor<D, T>(config.shape);

    // Initialize output.
    JST_INIT(
        JST_INIT_OUTPUT("coeffs", output.coeffs, config.shape);
    );

    // Calculate fractions.
    JST_CHECK(calculateFractions());

    // Generate all coefficients.
    JST_CHECK(generateSincFunction());
    JST_CHECK(generateWindow());
    JST_CHECK(generateUpconvert());

    // Create FFT plan for baking.
    auto inBuf = reinterpret_cast<fftwf_complex*>(scratchCoeffs.data());
    auto outBuf = reinterpret_cast<fftwf_complex*>(output.coeffs.data());

    const I32 M = config.shape[0];
    const I32 N = config.shape[1];

    int rank     = 1;      // Number of dimensions
    int n[]      = { N };  // Size of each dimension
    int howmany  = M;      // Number of FFTs
    int idist    = N;      // Distance between consecutive elements in input array
    int odist    = N;      // Distance between consecutive elements in output array
    int istride  = 1;      // Stride between successive elements in same FFT
    int ostride  = 1;      // Stride between successive elements in same FFT
    int *inembed = n;      // Pointer to array of dimensions for input
    int *onembed = n;      // Pointer to array of dimensions for output

    plan = fftwf_plan_many_dft(rank, n, howmany,
                               inBuf, inembed, istride, idist,
                               outBuf, onembed, ostride, odist,
                               FFTW_FORWARD, FFTW_ESTIMATE);

    if (!plan) {
        JST_ERROR("Failed to create FFT plan.");
        return Result::ERROR;
    }

    // Finalize filter.
    JST_CHECK(bakeFilter());

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Filter<D, T>::destroy() {
    fftwf_destroy_plan(plan);

    return Result::SUCCESS;
}

template<Device D, typename T>
void Filter<D, T>::summary() const {
    JST_INFO("  Signal Sample Rate: {}", config.signalSampleRate);
    JST_INFO("  Filter Sample Rate: {}", config.filterSampleRate);
    JST_INFO("  Filter Center:      {}", config.filterCenter);
    JST_INFO("  Number Of Taps:     {}", config.numberOfTaps);
    JST_INFO("  Linear Frequency:   {}", config.linearFrequency ? "YES" : "NO");
    JST_INFO("  Filter Shape:       {}", config.shape);
}

template class Filter<Device::CPU, CF32>;

}  // namespace Jetstream
