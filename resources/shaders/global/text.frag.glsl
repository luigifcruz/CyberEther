#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec3 color;
    float sharpness;
} uniforms;

layout(location = 0) in vec2 inTexcoord;
layout(location = 1) in vec4 inColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D remoteFramebufferTex;
layout(set = 0, binding = 2) uniform sampler remoteFramebufferSam;

void main() {
    float sampleValue = texture(
        sampler2D(remoteFramebufferTex, remoteFramebufferSam),
        inTexcoord
    ).r;

    // Matches src/render/components/font.cc.
    const float edge = 128.0 / 255.0;
    const float atlasPixelRange = 255.0 / 16.0;

    // Convert STB SDF to screen-pixel distance.
    vec2 atlasSize = vec2(textureSize(
        sampler2D(remoteFramebufferTex, remoteFramebufferSam),
        0
    ));

    vec2 uvFwidth = max(fwidth(inTexcoord), vec2(1.0e-6));
    vec2 screenTexSize = vec2(1.0) / uvFwidth;
    vec2 unitRange = vec2(atlasPixelRange) / atlasSize;
    float screenPixelRange = max(0.5 * dot(unitRange, screenTexSize), 1.0e-6);

    float screenPixelDistance = (sampleValue - edge) * screenPixelRange;

    // One-screen-pixel coverage ramp.
    float alpha = clamp(screenPixelDistance + 0.5, 0.0, 1.0);

    outColor = vec4(inColor.rgb, inColor.a * alpha);
}
