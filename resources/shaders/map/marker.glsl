#include "camera.glsl"

layout(location = 0) in vec2 inQuad;
// Instance: longitude, latitude, logical pixel radius, opacity.
layout(location = 1) in vec4 inMarker;

layout(location = 0) out vec2 vLocal;
layout(location = 1) out float vOpacity;

void main() {
    vec3 p = lonLatToSphere(inMarker.x, inMarker.y);
    if (dot(p, uniforms.targetNormal.xyz) < uniforms.targetNormal.w) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        vLocal = vec2(0.0);
        vOpacity = 0.0;
        return;
    }

    vLocal = inQuad;
    vOpacity = inMarker.w;
    gl_Position = vec4(projectToNdc(p) + inQuad * inMarker.z * mapPixelSize(), 0.0, 1.0);
}
