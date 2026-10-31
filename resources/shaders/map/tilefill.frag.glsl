#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec3 vSphere;
layout(location = 2) flat in vec2 vFade;

layout(location = 0) out vec4 outColor;

void main() {
    if (dot(normalize(vSphere), uniforms.targetNormal.xyz) <
        uniforms.targetNormal.w) {
        discard;
    }
    float alpha = uniforms.lineOpacity * tileFade(vFade);
    if (alpha <= 0.0) {
        discard;
    }
    outColor = vec4(vColor, alpha);
}
