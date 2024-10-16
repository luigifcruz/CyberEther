#ifndef JETSTREAM_MODULES_CONSTELLATION_HH
#define JETSTREAM_MODULES_CONSTELLATION_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_CONSTELLATION_CPU(MACRO) \
    MACRO(Constellation, CPU, CF32)

template<Device D, typename T = CF32>
class Constellation : public Module, public Compute, public Present {
 public:
    Constellation();
    ~Constellation();

    // Configuration 

    struct Config {
        Extent2D<U64> viewSize = {512, 512};

        JST_SERDES(viewSize);
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
        JST_SERDES_OUTPUT();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor
    
    Result create();

    // Miscellaneous

    constexpr const Extent2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Extent2D<U64>& viewSize(const Extent2D<U64>& viewSize);

    Render::Texture& getTexture();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    struct GImpl;
    std::unique_ptr<GImpl> gimpl;

    Tensor<D, F32> timeSamples;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;
    std::shared_ptr<Render::Buffer> signalUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Texture> signalTexture;
    std::shared_ptr<Render::Texture> lutTexture;

    std::shared_ptr<Render::Program> program;

    std::shared_ptr<Render::Surface> surface;

    std::shared_ptr<Render::Vertex> vertex;

    std::shared_ptr<Render::Draw> drawVertex;

    F32 decayFactor;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE
JST_CONSTELLATION_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
