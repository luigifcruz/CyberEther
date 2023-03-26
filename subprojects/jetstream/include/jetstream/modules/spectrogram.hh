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

template<Device D, typename IT = F32>
class Spectrogram : public Module, public Compute, public Present {
 public:
    struct Config {
        U64 height = 256;
        Render::Size2D<U64> viewSize = {4096, 512};
    };

    struct Input {
        const Vector<D, IT, 2>& buffer;
    };

    struct Output {
    };

    explicit Spectrogram(const Config& config,
                         const Input& input); 

    constexpr const Device device() const {
        return D;
    }

    constexpr const Taint taints() const {
        return Taint::None; 
    }

    void summary() const final;

    constexpr const Config getConfig() const {
        return config;
    }

    constexpr const Render::Size2D<U64>& viewSize() const {
        return config.viewSize;
    }
    const Render::Size2D<U64>& viewSize(const Render::Size2D<U64>& viewSize);

    Render::Texture& getTexture();

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
    std::shared_ptr<Render::Buffer> binTexture;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    const Result createCompute(const RuntimeMetadata& meta) final;
    const Result compute(const RuntimeMetadata& meta) final;

    const Result createPresent(Render::Window& window) final;
    const Result present(Render::Window& window) final;

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
    
    //
    // Metal
    //

    const char* MetalShader = R"END(
        #include <metal_stdlib>

        using namespace metal;

        struct TexturePipelineRasterizerData {
            float4 position [[position]];
            float2 texcoord;
            uint maxSize;
        };

        struct ShaderUniforms {
            uint width;
            uint height;
            float offset;
            float zoom;
        };

        vertex TexturePipelineRasterizerData vertFunc(
                constant ShaderUniforms& uniforms [[buffer(0)]],
                const device packed_float3* vertexArray [[buffer(2)]],
                const device packed_float2* texcoord [[buffer(3)]],
                unsigned int vID[[vertex_id]]) {
            TexturePipelineRasterizerData out;

            out.position = vector_float4(vertexArray[vID], 1.0);
            out.texcoord = float2(texcoord[vID].x * uniforms.width,
                                  texcoord[vID].y * uniforms.height);
            out.maxSize = uniforms.width * uniforms.height;

            return out;
        }

        fragment float4 fragFunc(
                TexturePipelineRasterizerData in [[stage_in]],
                constant ShaderUniforms& uniforms [[buffer(0)]],
                const device float* data [[buffer(1)]],
                texture2d<float> lut [[texture(0)]]) {
            uint2 texcoord = uint2(in.texcoord);
            uint index = texcoord.y * uniforms.width + texcoord.x;

            if (index > in.maxSize && index < 0) {
                return float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            constexpr sampler lutSampler(filter::linear);
            return lut.sample(lutSampler, vector_float2(data[index], 0.0));
        }
    )END";
};

}  // namespace Jetstream

#endif
