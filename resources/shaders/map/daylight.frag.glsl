#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"
#include "sun.glsl"

layout(location = 0) in vec3 vSphere;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(vSphere);
    if (dot(normal, uniforms.targetNormal.xyz) < uniforms.targetNormal.w) {
        discard;
    }

    float elevation = sunElevation(normal);
    float night = 1.0 - sunDaylight(normal);

    // Night sinks the cartography toward a deep blue-black; the day side gets
    // a faint neutral lift that peaks under the subsolar point.
    float shade = sun.night.x * night * sun.direction.w;
    vec3 shadeColor = vec3(0.010, 0.016, 0.045);
    float lift = clamp(sun.night.z * pow(max(elevation, 0.0), 0.75) *
                       sun.direction.w, 0.0, 1.0);
    vec3 liftColor = vec3(1.0);

    // Fold both translucent layers into one blend: shade first, lift on top.
    float alpha = 1.0 - (1.0 - shade) * (1.0 - lift);
    if (alpha < 0.002) {
        discard;
    }
    vec3 color = (shadeColor * shade * (1.0 - lift) + liftColor * lift) / alpha;
    outColor = vec4(color, alpha);
}
