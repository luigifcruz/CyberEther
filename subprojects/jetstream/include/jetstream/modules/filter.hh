#ifndef JETSTREAM_MODULES_FILTER_HH
#define JETSTREAM_MODULES_FILTER_HH

#include <fftw3.h>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

// TODO: Fix input/output template arguments.
template<Device D, typename T = CF32>
class Filter : public Module {
 public:
    struct Config {
        F32 signalSampleRate;
        F32 filterSampleRate = 1e6;
        F32 filterCenter = 0.0;
        VectorShape<2> shape;
        U64 numberOfTaps = 101;
        bool linearFrequency = true;
    };

    struct Input {
    };

    struct Output {
        Vector<D, T, 2> coeffs;
    };

    explicit Filter(const Config& config,
                    const Input& input);

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Filter";
    }

    void summary() const final;

    constexpr const Vector<D, T, 2>& getCoeffsBuffer() const {
        return this->output.coeffs;
    }

    constexpr const Config getConfig() const {
        return config;
    }

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

    // TODO: Copy only when compute.

    static Result Factory(std::unordered_map<std::string, std::any>& config,
                          std::unordered_map<std::string, std::any>& input,
                          std::unordered_map<std::string, std::any>& output,
                          std::shared_ptr<Filter<D, T>>& module);

 private:
    Config config;
    const Input input;
    Output output;

    fftwf_plan plan;

    Vector<D, typename T::value_type, 2> sincCoeffs;
    Vector<D, typename T::value_type, 2> windowCoeffs;
    Vector<D, T, 2> upconvertCoeffs;
    Vector<D, T, 2> scratchCoeffs;
    F32 filterWidth;
    F32 filterOffset;

    Result calculateFractions();
    Result generateSincFunction();
    Result generateWindow();
    Result generateUpconvert();
    Result bakeFilter();
};

}  // namespace Jetstream

#endif
