#ifndef JETSTREAM_MODULES_SCALE_HH
#define JETSTREAM_MODULES_SCALE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Scale : public Module, public Compute {
 public:
    struct Config {
        U64 size;
        Range<T> range = {-1.0, +1.0};
    };

    struct Input {
        // TODO: Modify back.
        const Vector<Device::CPU, T>& buffer;
    };

    struct Output {
        // TODO: Modify back.
        Vector<Device::CPU, T> buffer;
    };

    explicit Scale(const Config& config,
                   const Input& input);

    constexpr const Device device() const {
        return D;
    }

    constexpr const Taint taints() const {
        return Taint::None;
    }

    void summary() const final;

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    // TODO: Modify back.
    constexpr const Vector<Device::CPU, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return this->config;
    }

    constexpr const Range<T>& range() const {
        return this->config.range;
    }

    const Range<T>& range(const Range<T>& range) {
        this->config.range = range;
        return range;
    }

 protected:
    const Result createCompute(const RuntimeMetadata& meta) final;
    const Result compute(const RuntimeMetadata& meta) final;

 private:
    Config config;
    const Input input;
    Output output;

#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        float min;
        float max;
    };

    struct {
        MTL::ComputePipelineState* state;
        Vector<Device::CPU, U8> constants;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
