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
        float vertical = (texcoord[vID].y * 2000.0);
        float horizontal = (texcoord[vID].x * 65536.0);
        out.texcoord = vector_float2(horizontal, vertical);

        return out;
    }

    struct asd {
        float color[2000][65536];
    };

    fragment float4 fragFunc(
            TexturePipelineRasterizerData in [[stage_in]],
            const device asd* data [[buffer(0)]],
            texture2d<float> lut [[texture(0)]],
            constant uint32_t& interpolate [[buffer(28)]]) {
        float mag = 0.0;

        mag = data->color[(int)in.texcoord.y][(int)in.texcoord.x];

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
