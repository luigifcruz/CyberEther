#ifndef JETSTREAM_MODULES_SCALE_HH
#define JETSTREAM_MODULES_SCALE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Scale : public Module, public Compute {
 public:
    struct Config {
        Range<T> range = {-1.0, +1.0};
    };

    struct Input {
        const Vector<D, T, 2> buffer;
    };

    struct Output {
        Vector<D, T, 2> buffer;
    };

    explicit Scale(const Config& config,
                   const Input& input);

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Scale";
    }

    void summary() const final;

    constexpr const Vector<D, T, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr Config getConfig() const {
        return this->config;
    }

    constexpr const Range<T>& range() const {
        return this->config.range;
    }

    const Range<T>& range(const Range<T>& range) {
        this->config.range = range;
        return range;
    }

    static Result Factory(std::unordered_map<std::string, std::any>& config,
                          std::unordered_map<std::string, std::any>& input,
                          std::unordered_map<std::string, std::any>& output,
                          std::shared_ptr<Scale<D, T>>& module);

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
    Config config;
    const Input input;
    Output output;

#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        F32 min;
        F32 max;
    };

    struct {
        MTL::ComputePipelineState* state;
        Vector<Device::Metal, U8> constants;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
