#ifndef JETSTREAM_MODULES_FILTER_HH
#define JETSTREAM_MODULES_FILTER_HH

#include <fftw3.h>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"

namespace Jetstream {

// TODO: Copy coeffs only when compute.
template<Device D, typename T = CF32>
class Filter : public Module {
 public:
    // Configuration 

    struct Config {
        F32 signalSampleRate;
        F32 filterSampleRate = 1e6;
        F32 filterCenter = 0.0;
        std::vector<U64> shape;
        U64 numberOfTaps = 101;
        bool linearFrequency = true;

        JST_SERDES(
            JST_SERDES_VAL("signalSampleRate", signalSampleRate);
            JST_SERDES_VAL("filterSampleRate", filterSampleRate);
            JST_SERDES_VAL("filterCenter", filterCenter);
            JST_SERDES_VAL("shape", shape);
            JST_SERDES_VAL("numberOfTaps", numberOfTaps);
            JST_SERDES_VAL("linearFrequency", linearFrequency);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> coeffs;

        JST_SERDES(
            JST_SERDES_VAL("coeffs", coeffs);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputCoeffs() const {
        return this->output.coeffs;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "filter";
    }

    std::string_view prettyName() const {
        return "Filter";
    }

    void summary() const final;

    // Constructor

    Result create();
    Result destroy();

    // Miscellaneous

    constexpr const F32& filterSampleRate() const {
        return this->config.filterSampleRate;
    }

    const F32& filterSampleRate(const F32& sampleRate) {
        this->config.filterSampleRate = sampleRate;
        JST_CHECK_THROW(calculateFractions());
        JST_CHECK_THROW(generateSincFunction());
        JST_CHECK_THROW(bakeFilter());
        return sampleRate;
    }

    constexpr const F32& filterCenter() const {
        return this->config.filterCenter;
    }

    const F32& filterCenter(const F32& center) {
        this->config.filterCenter = center;
        JST_CHECK_THROW(calculateFractions());
        JST_CHECK_THROW(generateUpconvert());
        JST_CHECK_THROW(bakeFilter());
        return center;
    }

    constexpr const U64& filterTaps() const {
        return this->config.numberOfTaps;
    }

    const U64& filterTaps(const U64& numberOfTaps) {
        this->config.numberOfTaps = numberOfTaps;
        JST_CHECK_THROW(calculateFractions());
        JST_CHECK_THROW(generateSincFunction());
        JST_CHECK_THROW(generateWindow());
        JST_CHECK_THROW(generateUpconvert());
        JST_CHECK_THROW(bakeFilter());
        return numberOfTaps;
    }

 private:
    fftwf_plan plan;

    Tensor<D, typename T::value_type> sincCoeffs;
    Tensor<D, typename T::value_type> windowCoeffs;
    Tensor<D, T> upconvertCoeffs;
    Tensor<D, T> scratchCoeffs;
    F32 filterWidth;
    F32 filterOffset;

    Result calculateFractions();
    Result generateSincFunction();
    Result generateWindow();
    Result generateUpconvert();
    Result bakeFilter();

    JST_DEFINE_IO();
};

}  // namespace Jetstream

#endif
