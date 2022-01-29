#ifndef JSTCORE_WATERFALL_SHADERS_H
#define JSTCORE_WATERFALL_SHADERS_H

#include "render/base.hpp"
#include "render/extras.hpp"

namespace Jetstream::Waterfall {

//
// Metal
//

// TODO: Fix macro invalid memory access.
inline const char* MetalShader = R"END(
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

//
// GLES
//

inline const char* GlesVertexShader = R"END(#version 310 es
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    layout (std140, binding = 0) readonly buffer ShaderUniforms {
        int width;
        int height;
        int maxSize;
        float index;
        float offset;
        float zoom;
        bool interpolate;
    };

    void main() {
        gl_Position = vec4(aPos, 1.0);
        float vertical = (aTexCoord.y) * float(height);
        float horizontal = aTexCoord.x * float(width);
        TexCoord = vec2(horizontal, vertical);
    }
)END";

inline const char* GlesFragmentShader = R"END(#version 310 es
    precision highp float;

    out vec4 FragColor;
    in vec2 TexCoord;

    uniform sampler2D LutTexture;

    layout (std140, binding = 0) readonly buffer ShaderUniforms {
        int width;
        int height;
        int maxSize;
        float index;
        float offset;
        float zoom;
        bool interpolate;
    };

    layout (std140, binding = 1) readonly buffer WaterfallData {
        float data[];
    };

    float SAMPLER(float x, float y) {
        int _idx = ((int(y)) * width) + (int(x));

        if (_idx < maxSize && _idx > 0) {
            return data[_idx];
        }

        if (_idx < 0) {
            return data[_idx + maxSize];
        }

        return 1.0;
    }

    void main() {
        float mag = 0.0;

        mag = SAMPLER(TexCoord.x, TexCoord.y);

        FragColor = texture(LutTexture, vec2(mag, 0.0));
    }
)END";

} // namespace Jetstream::Waterfall

#endif
