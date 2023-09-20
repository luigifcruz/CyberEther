#ifndef JETSTREAM_MODULES_SPECTOGRAM_HH
#define JETSTREAM_MODULES_SPECTOGRAM_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Spectrogram : public Module, public Compute, public Present {
 public:
    // Configuration 

    struct Config {
        U64 height = 256;
        Size2D<U64> viewSize = {512, 384};

        JST_SERDES(
            JST_SERDES_VAL("height", height);
            JST_SERDES_VAL("viewSize", viewSize);
        );
    };
    
    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "spectrogram";
    }

    std::string_view prettyName() const {
        return "Spectrogram";
    }

    void summary() const final;
    
    // Constructor

    Result create();

    // Miscellaneous

    constexpr const Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Size2D<U64>& viewSize(const Size2D<U64>& viewSize);

    Render::Texture& getTexture();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

    Result createPresent() final;
    Result present() final;
    Result destroyPresent() final;

 private:
    struct {
        U32 width;
        U32 height;
        F32 offset;
        F32 zoom;
    } shaderUniforms;

    F32 decayFactor;
    Vector<D, F32, 2> frequencyBins;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> binTexture;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        U32 width;
        U32 height;
        F32 decayFactor;
        U32 batchSize;
    };

    struct {
        MTL::ComputePipelineState* stateDecay;
        MTL::ComputePipelineState* stateActivate;
        Vector<Device::Metal, U8> constants;
    } metal;
#endif

    JST_DEFINE_MODULE_IO();
};

}  // namespace Jetstream

#endif
