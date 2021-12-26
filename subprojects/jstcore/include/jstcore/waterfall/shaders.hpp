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
        float vertical = (index + texcoord[vID].y);
        float horizontal = (texcoord[vID].x / zoom) + offset;
        out.texcoord = vector_float2(horizontal, vertical);

        return out;
    }

    fragment float4 fragFunc(
            TexturePipelineRasterizerData in [[stage_in]],
            texture2d<float> data [[texture(0)]],
            texture2d<float> lut [[texture(1)]],
            constant uint32_t& interpolate [[buffer(28)]]) {
        float mag = 0.0;
        sampler lutSampler(filter::linear);
        constexpr sampler dataSampler(filter::linear, address::repeat);

        if (interpolate == 1) {
            const float yBlur = 1.0 / data.get_height();
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y - 4.0*yBlur)).r * 0.0162162162;
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y - 3.0*yBlur)).r * 0.0540540541;
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y - 2.0*yBlur)).r * 0.1216216216;
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y - 1.0*yBlur)).r * 0.1945945946;
            mag += data.sample(dataSampler, in.texcoord).r * 0.2270270270;
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y + 1.0*yBlur)).r * 0.1945945946;
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y + 2.0*yBlur)).r * 0.1216216216;
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y + 3.0*yBlur)).r * 0.0540540541;
            mag += data.sample(dataSampler, float2(in.texcoord.x, in.texcoord.y + 4.0*yBlur)).r * 0.0162162162;
        }

        if (interpolate == 0) {
            mag = data.sample(dataSampler, in.texcoord).r;
        }

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
    void main() {
        gl_Position = vec4(aPos, 1.0);
        float coord = (index-aTexCoord.y);
        TexCoord = vec2(aTexCoord.x, coord);
    }
)END";

inline const char* GlesFragmentShader = R"END(#version 300 es
    precision highp float;
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform int interpolate;
    uniform int drawIndex;
    uniform sampler2D BinTexture;
    uniform sampler2D LutTexture;
    vec4 cubic(float v){
        vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
        vec4 s = n * n * n;
        float x = s.x;
        float y = s.y - 4.0 * s.x;
        float z = s.z - 4.0 * s.y + 6.0 * s.x;
        float w = 6.0 - x - y - z;
        return vec4(x, y, z, w) * (1.0/6.0);
    }
    vec4 textureBicubic(sampler2D sampler, vec2 texCoords){
        vec2 texSize = vec2(textureSize(sampler, 0));
        vec2 invTexSize = 1.0 / texSize;
        texCoords = texCoords * texSize - 0.5;
        vec2 fxy = fract(texCoords);
        texCoords -= fxy;
        vec4 xcubic = cubic(fxy.x);
        vec4 ycubic = cubic(fxy.y);
        vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;
        vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
        vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;
        offset *= invTexSize.xxyy;
        vec4 sample0 = texture(sampler, offset.xz);
        vec4 sample1 = texture(sampler, offset.yz);
        vec4 sample2 = texture(sampler, offset.xw);
        vec4 sample3 = texture(sampler, offset.yw);
        float sx = s.x / (s.x + s.y);
        float sy = s.z / (s.z + s.w);
        return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
    }
    void main() {
        float mag;
        if (interpolate == 1) {
            mag = textureBicubic(BinTexture, TexCoord).r;
        }
        if (interpolate == 0) {
            mag = texture(BinTexture, TexCoord).r;
        }
        FragColor = texture(LutTexture, vec2(mag, 0));
    }
)END";

} // namespace Jetstream::Waterfall

#endif
