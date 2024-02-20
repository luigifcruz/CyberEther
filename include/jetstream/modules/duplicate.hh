#ifndef JETSTREAM_MODULES_DUPLICATE_HH
#define JETSTREAM_MODULES_DUPLICATE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_DUPLICATE_CPU(MACRO) \
    MACRO(Duplicate, CPU, CF32) \
    MACRO(Duplicate, CPU, F32)

#define JST_DUPLICATE_CUDA(MACRO) \
    MACRO(Duplicate, CUDA, CF32) \
    MACRO(Duplicate, CUDA, F32)

template<Device D, typename T = CF32>
class Duplicate : public Module, public Compute {
 public:
    Duplicate();
    ~Duplicate();

    // Configuration 

    struct Config {
        JST_SERDES();
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_DUPLICATE_CPU_AVAILABLE
JST_DUPLICATE_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_DUPLICATE_CUDA_AVAILABLE
JST_DUPLICATE_CUDA(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
