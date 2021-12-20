#ifndef JSTCORE_LINEPLOT_GENERIC_H
#define JSTCORE_LINEPLOT_GENERIC_H

#include "jetstream/base.hpp"
#include "render/tools/lut.hpp"
#include "render/base.hpp"

namespace Jetstream::Lineplot {

class Generic : public Module {
public:
    struct Config {
        std::shared_ptr<Render::Instance> render;
        Size2D<int> size {2500, 500};
    };

    struct Input {
        const Data<VF32> in;
    };

    explicit Generic(const Config&, const Input&);
    virtual ~Generic() = default;

    Result compute();
    Result present();

    constexpr Size2D<int> size() const {
        return config.size;
    }
    Size2D<int> size(const Size2D<int>&);

    std::weak_ptr<Render::Texture> tex() const;

protected:
    Config config;
    const Input input;

    std::vector<float> plot;
    std::vector<float> grid;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> lineVertex;
    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Draw> drawLineVertex;

    Result initRender(float*, bool cudaInterop = false);

    /*
    const char* vertexSource = R"END(#version 300 es
        layout (location = 0) in vec3 aPos;
        out vec2 A;
        void main() {
            float min_x = 0.0;
            float max_x = 1.0;
            float y = -(2.0 * ((aPos.y - min_x)/(max_x - min_x)) - 1.0);
            gl_Position = vec4(aPos.x, y, aPos.z, 1.0);
            float pos = (y + 1.0)/2.0;
            A = vec2(-pos, 0.0);
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

        struct RenderUniforms {
            uint32_t drawIndex;
        };

        vertex TexturePipelineRasterizerData vertFunc(
                const device packed_float3* vertexArray [[buffer(0)]],
                unsigned int vID[[vertex_id]]) {
            float3 vert = vertexArray[vID];
            TexturePipelineRasterizerData out;

            float min_x = 0.0;
            float max_x = 1.0;
            float y = (2.0 * ((vert.y - min_x)/(max_x - min_x)) - 1.0);
            out.position = vector_float4(vert.x, y, vert.z, 1.0);

            float pos = (y + 1.0)/2.0;
            out.texcoord = vector_float2(pos, 0.0);

            return out;
        }

        fragment float4 fragFunc(
                TexturePipelineRasterizerData in [[stage_in]],
                constant RenderUniforms& renderUniforms [[buffer(28)]],
                texture2d<float> lut [[texture(0)]]) {
            if (renderUniforms.drawIndex == 0) {
                return vector_float4(0.27, 0.27, 0.27, 1.0);
            }

            if (renderUniforms.drawIndex == 1) {
                sampler lutSampler(filter::linear);
                return lut.sample(lutSampler, in.texcoord);
            }

            return vector_float4(0.0, 1.0, 0.0, 1.0);
        }
    )END";

    const char* fragmentSource = R"END(#version 300 es
        precision highp float;
        out vec4 FragColor;
        in vec2 A;
        uniform int drawIndex;
        uniform sampler2D LutTexture;
        void main() {
            if (drawIndex == 0) {
                FragColor = vec4(0.27, 0.27, 0.27, 0.0);
            } else if (drawIndex == 1) {
                FragColor = texture(LutTexture, A);
            } else {
                FragColor = vec4(0.0, 1.0, 0.0, 0.0);
            }
        }
    )END";
};

} // namespace Jetstream::Lineplot

#endif
