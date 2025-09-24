#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/benchmark.hh"
#include "jetstream/render/base.hh"
#include "jetstream/memory2/tensor.hh"

namespace Jetstream {

class JETSTREAM_API Module {
 public:
    virtual ~Module() = default;

    virtual Result create() {
        return Result::SUCCESS;
    }

    virtual Result destroy() {
        return Result::SUCCESS;
    }

    virtual constexpr Taint taint() const {
        return Taint::CLEAN;
    }

    virtual void info() const = 0;
    virtual constexpr Device device() const = 0;

    constexpr const Locale& locale() const {
        return _locale;
    }

 protected:
    static Result InitInput(mem2::Tensor& buffer, const Taint& taint) {
        JST_TRACE("[MODULE] Init input tensor");

        if (buffer.empty()) {
            JST_ERROR("Input is empty during initialization.");
            return Result::ERROR;
        }

        if (!buffer.valid_shape()) {
            JST_ERROR("Input has invalid shape during initialization: {}", buffer.shape().size());
            return Result::ERROR;
        }

        if ((taint & Taint::DISCONTIGUOUS) != Taint::DISCONTIGUOUS && !buffer.contiguous()) {
            JST_ERROR("Input is not contiguous during initialization.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    static Result InitOutput(const std::string& name,
                             mem2::Tensor& buffer,
                             const Locale& locale) {
        JST_TRACE("[MODULE] Init output tensor: '{}'", name);

        if (locale.empty()) {
            JST_ERROR("Output locale is empty during initialization.");
            return Result::ERROR;
        }

        if (!buffer.empty()) {
            JST_ERROR("The output buffer should be empty during initialization.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    void setLocale(const Locale& locale) {
        _locale = locale;
    }

 private:
    Locale _locale;

    friend Instance;
};

class CPU;
class CUDA;
class Metal;

class JETSTREAM_API Compute {
 public:
    virtual ~Compute() = default;

    struct Context {
#ifdef JETSTREAM_GRAPH_CPU_AVAILABLE
        CPU* cpu;
#endif
#ifdef JETSTREAM_GRAPH_CUDA_AVAILABLE
        CUDA* cuda;
#endif
#ifdef JETSTREAM_GRAPH_METAL_AVAILABLE
        Metal* metal;
#endif
    };

    virtual constexpr Result createCompute(const Context&) {
        return Result::SUCCESS;
    }
    virtual constexpr Result compute(const Context&) {
        return Result::SUCCESS;
    }
    virtual constexpr Result computeReady() {
        return Result::SUCCESS;
    }
    virtual constexpr Result destroyCompute(const Context&) {
        return Result::SUCCESS;
    }

 protected:
    friend Instance;
};

class JETSTREAM_API Present {
 public:
    virtual ~Present() = default;

    virtual constexpr Result createPresent() {
        return Result::SUCCESS;
    }
    virtual constexpr Result present() {
        return Result::SUCCESS;
    }
    virtual constexpr Result destroyPresent() {
        return Result::SUCCESS;
    }

 protected:
    std::shared_ptr<Render::Window> window;

    friend Instance;
};

}  // namespace Jetstream

#endif
