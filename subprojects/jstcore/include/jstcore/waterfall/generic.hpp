#ifndef JSTCORE_WATERFALL_GENERIC_H
#define JSTCORE_WATERFALL_GENERIC_H

#include "jetstream/base.hpp"
#include "render/tools/lut.hpp"
#include "render/base.hpp"
#include "render/extras.hpp"

namespace Jetstream::Waterfall {

class Generic : public Module {
public:
    struct Config {
        std::shared_ptr<Render::Instance> render;
        bool interpolate = true;
        Size2D<int> size = {2500, 500};
    };

    struct Input {
        const Data<VF32> in;
    };

    explicit Generic(const Config&, const Input&);
    virtual ~Generic() = default;

    Result compute();
    Result present();

    constexpr bool interpolate() const {
        return config.interpolate;
    }
    bool interpolate(bool);

    constexpr Size2D<int> size() const {
        return config.size;
    }
    Size2D<int> size(const Size2D<int>&);

    std::weak_ptr<Render::Texture> tex() const;

protected:
    Config config;
    const Input input;

    int inc = 0, last = 0, ymax = 0;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> binTexture;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    Result initRender(uint8_t*, bool cudaInterop = false);

    /*
    const char* vertexSource = R"END(#version 300 es
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        uniform float Index;
        void main() {
            gl_Position = vec4(aPos, 1.0);
            float coord = (Index-aTexCoord.y);
            TexCoord = vec2(aTexCoord.x, coord);
        }
    )END";
    */

    const char* vertexSource = R"END(
        #include <metal_stdlib>

        using namespace metal;

        struct TexturePipelineRasterizerData {
            float4 position [[position]];
            float2 texcoord;
        };

        vertex TexturePipelineRasterizerData vertFunc(
            const device packed_float3* vertexArray [[buffer(0)]],
            const device packed_float2* texcoord [[buffer(1)]],
            unsigned int vID[[vertex_id]]) {
            TexturePipelineRasterizerData out;

            out.position = vector_float4(vertexArray[vID], 1.0);
            out.texcoord = texcoord[vID];

            return out;
        }

        fragment float4 fragFunc(
            TexturePipelineRasterizerData in [[stage_in]],
            texture2d<float> data [[texture(0)]],
            texture2d<float> bin [[texture(1)]]
        ) {
            sampler imgSampler;
            float4 colorSample = data.sample(imgSampler, in.texcoord);
            return colorSample;
        }
    )END";
};

} // namespace Jetstream::Waterfall

#endif
