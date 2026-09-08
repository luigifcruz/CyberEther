#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
    int width;
    int height;
    int channels;
    int useLut;
    int interpolate;
    float rangeMin;
    float rangeScale;
    float zoom;
    float centerX;
    float centerY;
    float fitScaleX;
    float fitScaleY;
    float paddingScaleX;
    float paddingScaleY;
} uniforms;

layout(set = 0, binding = 1) readonly buffer FrameBuffer {
    float data[];
};

layout(set = 0, binding = 2) uniform texture2D lutTex;
layout(set = 0, binding = 3) uniform sampler lutSam;

vec4 fetchTexel(int x, int y) {
    int base = ((y * uniforms.width) + x) * uniforms.channels;
    if (uniforms.channels == 1) {
        return vec4(data[base], 0.0, 0.0, 1.0);
    }
    float alpha = (uniforms.channels == 4) ? data[base + 3] : 1.0;
    return vec4(data[base], data[base + 1], data[base + 2], alpha);
}

vec4 sampleFrame(vec2 p) {
    vec2 texel = p * vec2(float(uniforms.width), float(uniforms.height));

    if (uniforms.interpolate == 0) {
        int x = clamp(int(texel.x), 0, uniforms.width - 1);
        int y = clamp(int(texel.y), 0, uniforms.height - 1);
        return fetchTexel(x, y);
    }

    vec2 f = texel - 0.5;
    vec2 f0 = floor(f);
    vec2 t = f - f0;

    int x0 = clamp(int(f0.x), 0, uniforms.width - 1);
    int x1 = clamp(int(f0.x) + 1, 0, uniforms.width - 1);
    int y0 = clamp(int(f0.y), 0, uniforms.height - 1);
    int y1 = clamp(int(f0.y) + 1, 0, uniforms.height - 1);

    vec4 c00 = fetchTexel(x0, y0);
    vec4 c10 = fetchTexel(x1, y0);
    vec4 c01 = fetchTexel(x0, y1);
    vec4 c11 = fetchTexel(x1, y1);

    return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

void main() {
    vec2 view = vec2(inTexcoord.x, 1.0 - inTexcoord.y);
    vec2 zoomed = (view - 0.5) / uniforms.zoom + vec2(uniforms.centerX, uniforms.centerY);
    vec2 p = (zoomed - 0.5) * vec2(uniforms.fitScaleX, uniforms.fitScaleY) + 0.5;

    if (p.x < 0.0 || p.y < 0.0 || p.x >= 1.0 || p.y >= 1.0) {
        discard;
    }

    vec4 texel = sampleFrame(p);

    if (uniforms.channels == 1) {
        float scalar = clamp((texel.r - uniforms.rangeMin) * uniforms.rangeScale, 0.0, 1.0);
        if (uniforms.useLut != 0) {
            outColor = texture(sampler2D(lutTex, lutSam), vec2(scalar, 0.0));
        } else {
            outColor = vec4(vec3(scalar), 1.0);
        }
        return;
    }

    vec3 color = clamp((texel.rgb - uniforms.rangeMin) * uniforms.rangeScale, 0.0, 1.0);
    outColor = vec4(color, clamp(texel.a, 0.0, 1.0));
}
