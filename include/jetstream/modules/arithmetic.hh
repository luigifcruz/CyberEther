#ifndef JETSTREAM_MODULES_ARITHMETIC_HH
#define JETSTREAM_MODULES_ARITHMETIC_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_ARITHMETIC_CPU(MACRO) \
    MACRO(Arithmetic, CPU, CF32) \
    MACRO(Arithmetic, CPU, F32)

#define JST_ARITHMETIC_CUDA(MACRO) \
    MACRO(Arithmetic, CUDA, CF32) \
    MACRO(Arithmetic, CUDA, F32)

JST_SERDES_ENUM(ArithmeticOp, Add, Sub, Mul, Div);

template<Device D, typename T = CF32>
class Arithmetic : public Module, public Compute {
 public:
    Arithmetic();
    ~Arithmetic();

    // Configuration 

    struct Config {
        ArithmeticOp operation = ArithmeticOp::Add;
        U64 axis = 0;

        JST_SERDES(operation, axis);
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

    constexpr Taint taint() const {
        return Taint::DISCONTIGUOUS;
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

    Tensor<D, T> broadcasted_output;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_ARITHMETIC_CPU_AVAILABLE
JST_ARITHMETIC_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_ARITHMETIC_CUDA_AVAILABLE
JST_ARITHMETIC_CUDA(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::ArithmeticOp> : ostream_formatter {};

#endif
