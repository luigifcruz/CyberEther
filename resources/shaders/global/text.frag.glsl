#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec3 color;
    float sharpness;
    float atlasPixelRange;
} uniforms;

layout(location = 0) in vec2 inTexcoord;
layout(location = 1) in vec4 inColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D remoteFramebufferTex;
layout(set = 0, binding = 2) uniform sampler remoteFramebufferSam;

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
            float signedDistance = (sampleValue - edge) * screenPixelRange;
            alpha += clamp(signedDistance * slope + 0.5, 0.0, 1.0);
        }
    }
    alpha /= float(taps * taps);

    outColor = vec4(inColor.rgb, inColor.a * alpha);
}
