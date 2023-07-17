#ifndef JETSTREAM_MODULES_SPECTOGRAM_HH
#define JETSTREAM_MODULES_SPECTOGRAM_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Spectrogram : public Module, public Compute, public Present {
 public:
    struct Config {
        U64 height = 256;
        Render::Size2D<U64> viewSize = {2048, 512};
    };

    struct Input {
        const Vector<D, T, 2> buffer;
    };

    struct Output {
    };

    explicit Spectrogram(const Config& config,
                         const Input& input); 

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Spectrogram";
    }

    void summary() const final;

    constexpr Config getConfig() const {
        return config;
    }

    constexpr const Render::Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Render::Size2D<U64>& viewSize(const Render::Size2D<U64>& viewSize);

    Render::Texture& getTexture();

    static Result Factory(std::unordered_map<std::string, std::any>& config,
                          std::unordered_map<std::string, std::any>& input,
                          std::unordered_map<std::string, std::any>& output,
                          std::shared_ptr<Spectrogram<D, T>>& module,
                          const bool& castFromString = false);

 protected:
    Config config;
    const Input input;
    Output output;

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

    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

    Result createPresent(Render::Window& window) final;
    Result present(Render::Window& window) final;

 private:
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
};

}  // namespace Jetstream

#endif
