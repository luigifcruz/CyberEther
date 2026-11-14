#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec2 translation;
    float sharpness;
    float atlasPixelRange;
    float weight;
    float emboss;
    float em;
    float backdrop;
} uniforms;

layout(location = 0) in vec2 inTexcoord;
layout(location = 1) in vec4 inColor;
layout(location = 2) in float inLineY;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D remoteFramebufferTex;
layout(set = 0, binding = 2) uniform sampler remoteFramebufferSam;

float signedDistanceAt(vec2 uv, float screenPixelRange) {
    float sampleValue = textureLod(sampler2D(remoteFramebufferTex, remoteFramebufferSam), uv, 0.0).r;
    return (sampleValue - 128.0 / 255.0) * screenPixelRange + uniforms.weight;
}

void main() {
    vec2 atlasSize = vec2(textureSize(
        sampler2D(remoteFramebufferTex, remoteFramebufferSam),
        0
    ));

    vec2 dx = dFdx(inTexcoord);
    vec2 dy = dFdy(inTexcoord);
    vec2 uvFwidth = max(abs(dx) + abs(dy), vec2(1.0e-6));
    vec2 screenTexSize = vec2(1.0) / uvFwidth;
    vec2 unitRange = vec2(uniforms.atlasPixelRange) / atlasSize;
    float screenPixelRange = max(0.5 * dot(unitRange, screenTexSize), 1.0e-6);

    const float edge = 128.0 / 255.0;
    const int taps = 2;
    const float step = 1.0 / float(taps);
    float slope = 2.0 * uniforms.sharpness * float(taps);

    float alpha = 0.0;
    for (int j = 0; j < taps; j++) {
        for (int i = 0; i < taps; i++) {
            vec2 offset = vec2((float(i) + 0.5) * step - 0.5,
                               (float(j) + 0.5) * step - 0.5);
            vec2 uv = inTexcoord + offset.x * dx + offset.y * dy;
            float sampleValue = textureLod(
                sampler2D(remoteFramebufferTex, remoteFramebufferSam),
                uv,
                0.0
            ).r;
            float signedDistance = (sampleValue - edge) * screenPixelRange + uniforms.weight;
            alpha += clamp(signedDistance * slope + 0.5, 0.0, 1.0);
        }
    }
    alpha /= float(taps * taps);

    float emPixels = uniforms.em * screenPixelRange / max(uniforms.atlasPixelRange, 1.0e-6);
    float bevel = uniforms.emboss * emPixels;
    if (bevel <= 0.0) {
        outColor = vec4(inColor.rgb, inColor.a * alpha);
        return;
    }

    float center = signedDistanceAt(inTexcoord, screenPixelRange);
    vec2 gradient = vec2(
        signedDistanceAt(inTexcoord + dx, screenPixelRange) - signedDistanceAt(inTexcoord - dx, screenPixelRange),
        signedDistanceAt(inTexcoord + dy, screenPixelRange) - signedDistanceAt(inTexcoord - dy, screenPixelRange));
    float strength = length(gradient);
    vec2 normal = gradient / max(strength, 1.0e-4);
    float facing = dot(normal, normalize(vec2(0.3, 1.0))) * smoothstep(0.6, 1.4, strength);

    float sheen = clamp((inLineY / max(emPixels, 1.0e-4) - 0.1) / 0.4, 0.0, 1.0);
    float light = smoothstep(0.3, 0.7, uniforms.backdrop);
    vec3 top = inColor.rgb * mix(1.12, 1.25, light);
    vec3 bottom = inColor.rgb * mix(0.8, 1.02, light);
    vec3 body = mix(top, bottom, smoothstep(0.0, 1.0, sheen));

    float ramp = 1.0 - smoothstep(0.0, bevel, center);
    body *= 1.0 + 0.3 * ramp * facing;

    float groove = smoothstep(0.8, 1.6, center) * (1.0 - smoothstep(1.6, 3.0, center));
    body *= 1.0 - 0.1 * groove;

    float lip = 1.0 - smoothstep(0.0, 1.2, center);
    vec3 glint = mix(inColor.rgb, vec3(1.0), mix(0.45, 0.35, light));
    body = mix(body, glint, mix(1.0, 0.9, light) * lip * mix(0.55, 0.95, smoothstep(-0.6, 0.8, facing)));
    body = mix(body, inColor.rgb * 0.7, 0.35 * light * lip * smoothstep(0.1, 0.8, -facing));

    float specular = pow(max(mix(facing, -facing, light), 0.0), 4.0);
    float shine = 1.0 - smoothstep(0.0, 2.0, center);
    float inner = smoothstep(1.0, 2.0, center) * (1.0 - smoothstep(2.0, 4.0, center));
    vec3 highlight = mix(mix(inColor.rgb, vec3(1.0), 0.5), inColor.rgb * 0.7, light);
    body = mix(body, highlight, mix(0.9, 0.6, light) * shine * specular);
    body += inColor.rgb * mix(0.35, -0.1, light) * inner * specular;

    float gloss = light * (1.0 - smoothstep(0.5, 3.0, center)) * smoothstep(0.2, 0.9, facing) * (1.0 - sheen);
    body = mix(body, mix(inColor.rgb, vec3(1.0), 0.6), 0.25 * gloss);

    outColor = vec4(body, inColor.a * alpha);
}
