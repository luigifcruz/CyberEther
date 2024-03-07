#ifndef JETSTREAM_MODULES_WATERFALL_HH
#define JETSTREAM_MODULES_WATERFALL_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_WATERFALL_CPU(MACRO) \
    MACRO(Waterfall, CPU, F32)

#define JST_WATERFALL_METAL(MACRO) \
    MACRO(Waterfall, Metal, F32)

#define JST_WATERFALL_CUDA(MACRO) \
    MACRO(Waterfall, CUDA, F32)

template<Device D, typename T = F32>
class Waterfall : public Module, public Compute, public Present {
 public:
    Waterfall();
    ~Waterfall();

    // Configuration 

    struct Config {
        F32 zoom = 1.0;
        I32 offset = 0;
        U64 height = 512;
        bool interpolate = true;
        Size2D<U64> viewSize = {512, 384};

        JST_SERDES(zoom, offset, height, interpolate, viewSize);
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

    constexpr const Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Size2D<U64>& viewSize(const Size2D<U64>& viewSize);

    constexpr const bool& interpolate() const {
        return config.interpolate;
    }
    const bool& interpolate(const bool& interpolate);

    constexpr const F32& zoom() const {
        return config.zoom;
    }
    const F32& zoom(const F32& zoom);

    constexpr const I32& offset() const {
        return config.offset;
    }
    const I32& offset(const I32& offset);

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

    struct {
        int width;
        int height;
        int maxSize;
        float index;
        float offset;
        float zoom;
        bool interpolate;
    } shaderUniforms;

    U64 numberOfElements = 0;
    U64 numberOfBatches = 0;

    int inc = 0, last = 0, ymax = 0;
    Tensor<D, F32> frequencyBins;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Buffer> binTexture;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    Result underlyingCompute(const Context& ctx);

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE
JST_WATERFALL_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_CUDA_AVAILABLE
JST_WATERFALL_CUDA(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE
JST_WATERFALL_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
