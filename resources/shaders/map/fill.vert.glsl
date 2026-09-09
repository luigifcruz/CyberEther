#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

// Per-vertex: (longitude, latitude, r, g, b).
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec3 vSphere;

void main() {
    vec3 p = lonLatToSphere(inPosition.x, inPosition.y);
    // Project the real sphere point (no vertex-level clamping); the fragment
    // stage discards pixels beyond the horizon so triangles that straddle the
    // limb meet the silhouette cleanly without distorting large primitives.
    gl_Position = uniforms.viewProjection * vec4(p, 1.0);
    vColor = inColor;
    vSphere = p;
}
