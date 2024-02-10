#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/benchmark.hh"
#include "jetstream/render/base.hh"
#include "jetstream/memory/base.hh"

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

    virtual void info() const = 0;
    virtual constexpr Device device() const = 0;

    constexpr const Locale& locale() const {
        return _locale;
    }

 protected:
    template<Device DeviceId, typename DataType>
    static Result InitInput(Tensor<DeviceId, DataType>& buffer) {
        JST_TRACE("[MODULE] Init input locale: '{}'", buffer.locale());

        if (buffer.empty()) {
            JST_ERROR("Input is empty during initialization.");
            return Result::ERROR;
        }

        if (buffer.locale().empty()) {
            JST_ERROR("Input locale is empty during initialization.");
            return Result::ERROR;
        }

        if (!buffer.valid_shape()) {
            JST_ERROR("Input has invalid shape during initialization: {}", buffer.shape());
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<Device DeviceId, typename DataType>
    static Result InitOutput(const std::string& name, 
                             Tensor<DeviceId, DataType>& buffer,
                             const Locale& locale) {
        JST_TRACE("[MODULE] Init output locale: '{}'", locale);

        if (locale.empty()) {
            JST_ERROR("Output locale is empty during initialization.");
            return Result::ERROR;
        }
            
        buffer.set_locale({
            locale.blockId, 
            locale.moduleId, 
            name
        });

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

    virtual constexpr Result createCompute(const Context& ctx) {
        return Result::SUCCESS;
    }
    virtual constexpr Result compute(const Context& ctx) = 0;
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
    virtual constexpr Result present() = 0;
    virtual constexpr Result destroyPresent() {
        return Result::SUCCESS;
    }

 protected:
    std::shared_ptr<Render::Window> window;

    friend Instance;
};

}  // namespace Jetstream

#endif
