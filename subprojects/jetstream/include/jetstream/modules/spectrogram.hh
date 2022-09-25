#ifndef JETSTREAM_MODULES_SPECTOGRAM_HH
#define JETSTREAM_MODULES_SPECTOGRAM_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/render/extras.hh"

namespace Jetstream {

template<Device D, typename IT = F32>
class Spectrogram : public Module {
 public:
    struct Config {
        Render::Size2D<U64> viewSize = {4096, 512};
    };

    struct Input {
        const Memory::Vector<D, IT>& buffer;
    };

    struct Output {
    };

    explicit Spectrogram(const Config&, const Input&);

    constexpr const U64 getBufferSize() const {
        return input.buffer.size();
    }

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
        int width;
        int height;
        int maxSize;
        float index;
        float offset;
        float zoom;
        bool interpolate;
    } shaderUniforms;

    int inc = 0, last = 0, ymax = 0;
    std::vector<float> intermediate; 
    std::vector<float> frequencyBins;

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

    const Result compute(const RuntimeMetadata& meta = {}) final;
    const Result present(const RuntimeMetadata& meta = {}) final;

 private:
    const Result initializeRender();

    const Result underlyingInitialize();
    const Result underlyingCompute(const RuntimeMetadata& meta = {});

    //
    // Metal
    //

    // TODO: Improve gaussian blur.
    // TODO: Cleanup waterfall old code.
    // TODO: Fix macro invalid memory access.
    const char* MetalShader = R"END(
        #include <metal_stdlib>

        #define SAMPLER(x, y) ({ \
            int _idx = ((int)y)*uniforms.width+((int)x); \
            (_idx < uniforms.maxSize && _idx > 0) ? data[_idx] : \
                _idx += uniforms.maxSize; \
                (_idx < uniforms.maxSize && _idx > 0) ? data[_idx] : 1.0; })

        using namespace metal;

        struct TexturePipelineRasterizerData {
            float4 position [[position]];
            float2 texcoord;
        };

        struct ShaderUniforms {
            int width;
            int height;
            int maxSize;
            float index;
            float offset;
            float zoom;
            bool interpolate;
        };

        vertex TexturePipelineRasterizerData vertFunc(
                constant ShaderUniforms& uniforms [[buffer(0)]],
                const device packed_float3* vertexArray [[buffer(2)]],
                const device packed_float2* texcoord [[buffer(3)]],
                unsigned int vID[[vertex_id]]) {
            TexturePipelineRasterizerData out;

            out.position = vector_float4(vertexArray[vID], 1.0);
            float vertical = ((uniforms.index - (1.0 - texcoord[vID].y)) * (float)uniforms.height);
            float horizontal = (((texcoord[vID].x / uniforms.zoom) + uniforms.offset) * (float)uniforms.width);
            out.texcoord = float2(horizontal, vertical);

            return out;
        }

        fragment float4 fragFunc(
                TexturePipelineRasterizerData in [[stage_in]],
                constant ShaderUniforms& uniforms [[buffer(0)]],
                const device float* data [[buffer(1)]],
                texture2d<float> lut [[texture(0)]]) {
            float mag = 0.0;

            if (uniforms.interpolate) {
                mag += SAMPLER(in.texcoord.x, in.texcoord.y - 4.0) * 0.0162162162;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y - 3.0) * 0.0540540541;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y - 2.0) * 0.1216216216;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y - 1.0) * 0.1945945946;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y      ) * 0.2270270270;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y + 1.0) * 0.1945945946;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y + 2.0) * 0.1216216216;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y + 3.0) * 0.0540540541;
                mag += SAMPLER(in.texcoord.x, in.texcoord.y + 4.0) * 0.0162162162;
            } else {
                mag = SAMPLER(in.texcoord.x, in.texcoord.y);
            }

            constexpr sampler lutSampler(filter::linear);
            return lut.sample(lutSampler, vector_float2(mag, 0.0));
        }
    )END";
};

}  // namespace Jetstream

#endif
