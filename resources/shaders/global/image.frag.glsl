#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 frame;
    vec4 rect;
    vec4 params;
} uniforms;

layout(location = 0) in vec2 inTexcoord;
layout(location = 1) in vec2 inLocal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D imageTex;
layout(set = 0, binding = 2) uniform sampler imageSam;

void main() {
    vec4 color = texture(sampler2D(imageTex, imageSam), inTexcoord);
    if (uniforms.params.z > 0.5) {
        color.rgb = color.a > 0.0 ? color.rgb / color.a : vec3(0.0);
    }

    vec2 halfSize = uniforms.rect.zw * 0.5;
    float radius = clamp(uniforms.params.x, 0.0, min(halfSize.x, halfSize.y));
    vec2 q = abs(inLocal - halfSize) - (halfSize - vec2(radius));
    float distance = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    float coverage = clamp(0.5 - distance, 0.0, 1.0);

    outColor = vec4(color.rgb, color.a * coverage * uniforms.params.y);
}
