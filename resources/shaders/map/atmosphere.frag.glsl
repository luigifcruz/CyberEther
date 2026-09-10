#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"
#include "sun.glsl"

layout(location = 0) in vec3 vWorld;

layout(location = 0) out vec4 outColor;

void main() {
    float atmosphereRadius = uniforms.lineWidth;
    vec3 shellDirection = normalize(vWorld);
    float facing = dot(shellDirection, uniforms.targetNormal.xyz);
    float shellHorizon = atmosphereRadius * uniforms.targetNormal.w;
    if (facing < shellHorizon) {
        discard;
    }

    // The ray's closest approach to the globe center is 1.0 at the planet
    // silhouette and atmosphereRadius at the outer edge of the shell.
    vec3 ray = normalize(vWorld - uniforms.cameraPos.xyz);
    float alongRay = dot(-uniforms.cameraPos.xyz, ray);
    float radius = sqrt(max(
        dot(uniforms.cameraPos.xyz, uniforms.cameraPos.xyz) -
        alongRay * alongRay,
        0.0
    ));

    float innerGlow = pow(clamp(radius, 0.0, 1.0), 8.0);
    float outerGlow = pow(clamp(
        (atmosphereRadius - radius) / (atmosphereRadius - 1.0),
        0.0,
        1.0
    ), 1.5);
    outerGlow *= step(1.0, radius);

    // A narrow bright band anchors the softer glow to the physical limb.
    float outline = 1.0 - smoothstep(0.001, 0.004, abs(radius - 1.0));
    float haze = mix(innerGlow * uniforms.dashScale,
                     outerGlow * uniforms.dashPhase,
                     step(1.0, radius));
    // The shell only scatters where the sun reaches it, leaving a faint
    // airglow on the night limb.
    float sunlit = mix(1.0, 0.22 + 0.78 * sunDaylight(shellDirection),
                       sun.direction.w);
    float alpha = clamp((haze + outline * uniforms.outlineStrength) * sunlit,
                        0.0, 0.90);

    vec3 glowColor = vec3(uniforms.colorR, uniforms.colorG, uniforms.colorB);
    vec3 outlineColor = vec3(0.68, 0.84, 1.00);
    vec3 color = mix(glowColor, outlineColor,
                     clamp(outline * 0.75, 0.0, 1.0));
    outColor = vec4(color, alpha);
}
