#ifndef JSTCORE_WATERFALL_SHADERS_H
#define JSTCORE_WATERFALL_SHADERS_H

#include "render/base.hpp"
#include "render/extras.hpp"

namespace Jetstream::Waterfall {

//
// Metal
//

inline const char* MetalShader = R"END(
    #include <metal_stdlib>

    #define SAMPLER(x, y) ({ int _idx = ((int)y)*width+((int)x); (_idx < size && _idx > 0) ? data[_idx] : 0.0; })

    using namespace metal;

    struct TexturePipelineRasterizerData {
        float4 position [[position]];
        float2 texcoord;
    };

    vertex TexturePipelineRasterizerData vertFunc(
            const device packed_float3* vertexArray [[buffer(0)]],
            const device packed_float2* texcoord [[buffer(1)]],
            constant float& index [[buffer(29)]],
            constant float& zoom [[buffer(27)]],
            constant float& offset [[buffer(26)]],
            unsigned int vID[[vertex_id]]) {
        TexturePipelineRasterizerData out;

        out.position = vector_float4(vertexArray[vID], 1.0);
        float vertical = ((texcoord[vID].y) * 2000.0);
        float horizontal = (((texcoord[vID].x / zoom) + offset) * 65536.0);
        out.texcoord = float2(horizontal, vertical);

        return out;
    }

    fragment float4 fragFunc(
            TexturePipelineRasterizerData in [[stage_in]],
            const device float* data [[buffer(0)]],
            texture2d<float> lut [[texture(0)]],
            constant uint32_t& interpolate [[buffer(28)]]) {
        float mag = 0.0;

        const int width = 65536;
        const int size = (width * 2000);

        if (interpolate == 1) {
            mag += SAMPLER(in.texcoord.x, in.texcoord.y - 4.0) * 0.0162162162;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y - 3.0) * 0.0540540541;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y - 2.0) * 0.1216216216;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y - 1.0) * 0.1945945946;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y      ) * 0.2270270270;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y + 1.0) * 0.1945945946;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y + 2.0) * 0.1216216216;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y + 3.0) * 0.0540540541;
            mag += SAMPLER(in.texcoord.x, in.texcoord.y + 4.0) * 0.0162162162;
        }

        if (interpolate == 0) {
            mag = SAMPLER(in.texcoord.x, in.texcoord.y);
        }

        constexpr sampler lutSampler(filter::linear);
        return lut.sample(lutSampler, vector_float2(mag, 0.0));
    }
)END";

//
// GLES
//

inline const char* GlesVertexShader = R"END(#version 300 es
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    uniform float index;
    uniform float zoom;
    uniform float offset;

    void main() {
        gl_Position = vec4(aPos, 1.0);

        float vertical = (index - aTexCoord.y);
        float horizontal = (aTexCoord.x / zoom) + offset;
        TexCoord = vec2(horizontal, vertical);
    }
)END";

inline const char* GlesFragmentShader = R"END(#version 300 es
    precision highp float;

    out vec4 FragColor;
    in vec2 TexCoord;
    uniform int interpolate;
    uniform sampler2D BinTexture;
    uniform sampler2D LutTexture;

    void main() {
        float mag = 0.0;

        if (interpolate == 1) {
            const float yBlur = 1.0 / data.get_height();
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y - 4.0*yBlur)).r * 0.0162162162;
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y - 3.0*yBlur)).r * 0.0540540541;
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y - 2.0*yBlur)).r * 0.1216216216;
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y - 1.0*yBlur)).r * 0.1945945946;
            mag += texture(BinTexture, TexCoord).r * 0.2270270270;
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y + 1.0*yBlur)).r * 0.1945945946;
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y + 2.0*yBlur)).r * 0.1216216216;
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y + 3.0*yBlur)).r * 0.0540540541;
            mag += texture(BinTexture, vec2(TexCoord.x, TexCoord.y + 4.0*yBlur)).r * 0.0162162162;
        }

        if (interpolate == 0) {
            mag = texture(BinTexture, TexCoord).r;
        }

        return texture(LutTexture, vec2(mag, 0.0));
    }
)END";

} // namespace Jetstream::Waterfall

#endif
