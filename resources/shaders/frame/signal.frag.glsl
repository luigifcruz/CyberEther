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

vec4 fetchTexelClamped(int x, int y) {
    x = clamp(x, 0, uniforms.width - 1);
    y = clamp(y, 0, uniforms.height - 1);
    return fetchTexel(x, y);
}

vec4 cubicHermite(vec4 A, vec4 B, vec4 C, vec4 D, float t) {
    vec4 a = -A / 2.0 + (3.0 * B) / 2.0 - (3.0 * C) / 2.0 + D / 2.0;
    vec4 b = A - (5.0 * B) / 2.0 + 2.0 * C - D / 2.0;
    vec4 c = -A / 2.0 + C / 2.0;
    vec4 d = B;

    return a * t * t * t + b * t * t + c * t + d;
}

vec4 sampleNearest(vec2 p) {
    vec2 texel = p * vec2(float(uniforms.width), float(uniforms.height));
    int x = clamp(int(texel.x), 0, uniforms.width - 1);
    int y = clamp(int(texel.y), 0, uniforms.height - 1);
    return fetchTexel(x, y);
}

vec4 sampleBilinear(vec2 p) {
    vec2 texel = p * vec2(float(uniforms.width), float(uniforms.height));
    vec2 f = texel - 0.5;
    vec2 f0 = floor(f);
    vec2 t = f - f0;

    int x0 = clamp(int(f0.x), 0, uniforms.width - 1);
    int x1 = clamp(int(f0.x) + 1, 0, uniforms.width - 1);
    int y0 = clamp(int(f0.y), 0, uniforms.height - 1);
    int y1 = clamp(int(f0.y) + 1, 0, uniforms.height - 1);

    return mix(mix(fetchTexel(x0, y0), fetchTexel(x1, y0), t.x),
               mix(fetchTexel(x0, y1), fetchTexel(x1, y1), t.x), t.y);
}

vec4 sampleBicubic(vec2 p) {
    vec2 texel = p * vec2(float(uniforms.width), float(uniforms.height));
    vec2 f = texel - 0.5;
    vec2 f0 = floor(f);
    vec2 t = f - f0;

    int x = int(f0.x);
    int y = int(f0.y);

    vec4 rows[4];
    for (int j = 0; j < 4; ++j) {
        rows[j] = cubicHermite(
            fetchTexelClamped(x - 1, y - 1 + j),
            fetchTexelClamped(x,     y - 1 + j),
            fetchTexelClamped(x + 1, y - 1 + j),
            fetchTexelClamped(x + 2, y - 1 + j),
            t.x);
    }

    return cubicHermite(rows[0], rows[1], rows[2], rows[3], t.y);
}

vec4 sampleFrame(vec2 p) {
    if (uniforms.interpolate == 0) {
        return sampleNearest(p);
    }

    if (uniforms.interpolate == 2) {
        return sampleBicubic(p);
    }

    return sampleBilinear(p);
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
