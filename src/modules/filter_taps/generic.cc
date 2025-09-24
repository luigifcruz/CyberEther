#include "jetstream/modules/filter_taps.hh"

namespace Jetstream {

inline F64 sinc(F64 x) {
    return (x == 0) ? 1.0 : sin(JST_PI * x) / (JST_PI * x);
}

inline F64 n(U64 len, U64 index) {
    return static_cast<F64>(index - static_cast<F64>(len - 1) / 2.0);
}

template<Device D, typename T>
struct FilterTaps<D, T>::Impl {
    struct {
        U32 width;
        U32 height;
        F32 offset;
        F32 zoom;
    } signalUniforms;

    Tensor<D, typename T::value_type> sincCoeffs;
    Tensor<D, typename T::value_type> windowCoeffs;
    Tensor<D, T> upconvertCoeffs;

    bool baked = false;

    Result generateSincFunction(FilterTaps<D, T>& m);
    Result generateWindow(FilterTaps<D, T>& m);
    Result generateUpconvert(FilterTaps<D, T>& m);
};

template<Device D, typename T>
FilterTaps<D, T>::FilterTaps() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
FilterTaps<D, T>::~FilterTaps() {
    impl.reset();
}
template<Device D, typename T>
Result FilterTaps<D, T>::Impl::generateSincFunction(FilterTaps<D, T>& m) {
    const F64 filterWidth = (m.config.bandwidth / m.config.sampleRate) / 2.0;

    for (U64 i = 0; i < m.config.taps; i++) {
        sincCoeffs[i] = sinc(2.0 * filterWidth * (i - (m.config.taps - 1) / 2.0));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::Impl::generateWindow(FilterTaps<D, T>& m) {
    for (U64 i = 0; i < m.config.taps; i++) {
        windowCoeffs[i] = 0.42 - 0.50 * cos(2.0 * JST_PI * i / (m.config.taps - 1)) +
                          0.08 * cos(4.0 * JST_PI * i / (m.config.taps - 1));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::Impl::generateUpconvert(FilterTaps<D, T>& m) {
    const std::complex<F64> j(0.0, 1.0);

    for (U64 c = 0; c < m.config.center.size(); c++) {
        const F64 filterOffset = (m.config.center[c] / (m.config.sampleRate / 2.0)) / 2.0;

        for (U64 i = 0; i < m.config.taps; i++) {
            upconvertCoeffs[{c, i}] = std::exp(j * 2.0 * JST_PI * n(m.config.taps, i) * filterOffset);
        }
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::sampleRate(F32& sampleRate) {
    if (sampleRate < 0) {
        JST_ERROR("Invalid signal sample rate: {} MHz. Signal sample "
                  "rate should be positive.", sampleRate / JST_MHZ);
        sampleRate = config.sampleRate;
        return Result::ERROR;
    }
    config.sampleRate = sampleRate;
    output.coeffs.attribute("sample_rate").set(config.sampleRate);
    impl->baked = false;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::bandwidth(F32& bandwith) {
    if (bandwith < 0 or bandwith > config.sampleRate) {
        JST_ERROR("Invalid filter sample rate: {} MHz. Filter sample rate "
                  "should be between 0 MHz and {} MHz.", bandwith / JST_MHZ,
                                                         config.sampleRate / JST_MHZ);
        bandwith = config.bandwidth;
        return Result::ERROR;
    }
    config.bandwidth = bandwith;
    output.coeffs.attribute("bandwidth").set(config.bandwidth);
    impl->baked = false;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::center(const U64& idx, F32& center) {
    const F32 halfSampleRate = config.sampleRate / 2.0;
    if (center > halfSampleRate or center < -halfSampleRate) {
        JST_ERROR("Invalid center frequency: {} MHz. Center frequency should "
                  "be between {} MHz and {} MHz.", center / JST_MHZ,
                                                   -halfSampleRate / JST_MHZ,
                                                   +halfSampleRate / JST_MHZ);
        center = config.center[idx];
        return Result::ERROR;
    }
    config.center[idx] = center;
    output.coeffs.attribute("center").set(config.center);
    impl->baked = false;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::taps(U64& taps) {
    if ((taps % 2) == 0) {
        JST_ERROR("Invalid number of taps: '{}'. Number of taps should be odd.", taps);
        taps = config.taps;
        return Result::ERROR;
    }
    config.taps = taps;
    impl->baked = false;
    return Result::RELOAD;
}

template<Device D, typename T>
Result FilterTaps<D, T>::compute(const Context&) {
    if (impl->baked) {
        return Result::SUCCESS;
    }

    // Generate all coefficients.

    JST_CHECK(impl->generateSincFunction(*this));
    JST_CHECK(impl->generateWindow(*this));
    JST_CHECK(impl->generateUpconvert(*this));

    // Merge all coefficients.

    for (U64 c = 0; c < config.center.size(); c++) {
        for (U64 i = 0; i < config.taps; i++) {
            output.coeffs[{c, i}] = impl->sincCoeffs[{i}] *
                                    impl->windowCoeffs[{i}] *
                                    impl->upconvertCoeffs[{c, i}];
        }
    }

    impl->baked = true;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::create() {
    JST_DEBUG("Initializing Filter Taps module.");
    JST_INIT_IO();

    // Allocate internal data.

    JST_CHECK(impl->sincCoeffs.create(D, mem2::TypeToDataType<typename T::value_type>(), {config.taps}));
    JST_CHECK(impl->windowCoeffs.create(D, mem2::TypeToDataType<typename T::value_type>(), {config.taps}));
    JST_CHECK(impl->upconvertCoeffs.create(D, mem2::TypeToDataType<T>(), {config.center.size(), config.taps}));

    // Allocate output.

    JST_CHECK(output.coeffs.create(D, mem2::TypeToDataType<T>(), {config.center.size(), config.taps}));

    // Store parameters.

    output.coeffs.attribute("sample_rate").set(config.sampleRate);
    output.coeffs.attribute("bandwidth").set(config.bandwidth);
    output.coeffs.attribute("center").set(config.center);

    // Load parameters.

    JST_CHECK(this->taps(config.taps));
    JST_CHECK(this->sampleRate(config.sampleRate));
    JST_CHECK(this->bandwidth(config.bandwidth));
    for (U64 i = 0; i < config.center.size(); i++) {
        JST_CHECK(this->center(i, config.center[i]));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result FilterTaps<D, T>::destroy() {
    return Result::SUCCESS;
}

template<Device D, typename T>
void FilterTaps<D, T>::info() const {
    JST_DEBUG("  Signal Sample Rate: {}", config.sampleRate);
    JST_DEBUG("  Filter Sample Rate: {}", config.bandwidth);
    JST_DEBUG("  Filter Center:      {}", config.center);
    JST_DEBUG("  Number Of Taps:     {}", config.taps);
}

JST_FILTER_TAPS_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
