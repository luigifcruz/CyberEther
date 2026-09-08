#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec3 vSphere;

layout(location = 0) out vec4 outColor;

void main() {
    // Drop fragments on the far side of the globe (beyond the horizon).
    if (dot(normalize(vSphere), uniforms.targetNormal.xyz) <
        uniforms.targetNormal.w) {
        discard;
    }
    outColor = vec4(vColor, 1.0);
}
