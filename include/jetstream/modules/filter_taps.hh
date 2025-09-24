#ifndef JETSTREAM_MODULES_FILTER_TAPS_HH
#define JETSTREAM_MODULES_FILTER_TAPS_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory2/tensor.hh"

namespace Jetstream {

#define JST_FILTER_TAPS_CPU(MACRO) \
    MACRO(FilterTaps, CPU, CF32)

template<Device D, typename T = CF32>
class FilterTaps : public Module, public Compute {
 public:
    FilterTaps();
    ~FilterTaps();

    // Configuration

    struct Config {
        std::vector<F32> center = {0.0e6f};
        F32 sampleRate = 2.0e6f;
        F32 bandwidth = 1.0e6f;
        U64 taps = 101;

        JST_SERDES(center, sampleRate, bandwidth, taps);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES_INPUT();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor coeffs;

        JST_SERDES_OUTPUT(coeffs);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputCoeffs() const {
        return output.coeffs;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();
    Result destroy();

    // Miscellaneous

    constexpr const F32& sampleRate() const {
        return config.sampleRate;
    }
    Result sampleRate(F32& sampleRate);

    constexpr const F32& bandwidth() const {
        return config.bandwidth;
    }
    Result bandwidth(F32& bandwith);

    constexpr const F32& center(const U64& idx) const {
        return config.center[idx];
    }
    Result center(const U64& idx, F32& center);

    constexpr const U64& taps() const {
        return config.taps;
    }
    Result taps(U64& taps);

 protected:
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_FILTER_TAPS_CPU_AVAILABLE
JST_FILTER_TAPS_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
