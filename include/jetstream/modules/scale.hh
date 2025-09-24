#ifndef JETSTREAM_MODULES_SCALE_HH
#define JETSTREAM_MODULES_SCALE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_SCALE_CPU(MACRO) \
    MACRO(Scale, CPU, F32)

#define JST_SCALE_METAL(MACRO) \
    MACRO(Scale, Metal, F32)

#define JST_SCALE_CUDA(MACRO) \
    MACRO(Scale, CUDA, F32)

template<Device D, typename T = F32>
class Scale : public Module, public Compute {
 public:
    Scale();
    ~Scale();

    // Configuration

    struct Config {
        Range<T> range = {-1.0, +1.0};

        JST_SERDES(range);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();

    // Miscellaneous

    constexpr const Range<T>& range() const {
        return this->config.range;
    }

    const Range<T>& range(const Range<T>& range);

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
JST_SCALE_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_SCALE_CUDA_AVAILABLE
JST_SCALE_CUDA(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_SCALE_METAL_AVAILABLE
JST_SCALE_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
