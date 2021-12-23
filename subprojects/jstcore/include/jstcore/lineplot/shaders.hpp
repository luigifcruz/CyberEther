#ifndef JSTCORE_LINEPLOT_SHADERS_H
#define JSTCORE_LINEPLOT_SHADERS_H

#include "render/base.hpp"
#include "render/extras.hpp"

namespace Jetstream::Lineplot {

//
// Metal
//

inline const char* MetalShader = R"END(
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
            constant uint32_t& drawIndex [[buffer(30)]],
            texture2d<float> lut [[texture(0)]]) {
        if (drawIndex == 0) {
            return vector_float4(0.27, 0.27, 0.27, 1.0);
        }

        if (drawIndex == 1) {
            sampler lutSampler(filter::linear);
            return lut.sample(lutSampler, in.texcoord);
        }

        return vector_float4(0.0, 1.0, 0.0, 1.0);
    }
)END";

//
// GLES
//

inline const char* GlesVertexShader = R"END(#version 300 es
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

inline const char* GlesFragmentShader = R"END(#version 300 es
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

} // namespace Jetstream::Lineplot

#endif
