#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vLocal;
layout(location = 1) in float vIntensity;
layout(location = 2) in float vSeed;

layout(location = 0) out vec4 outColor;

const float HaloScale = 3.5;

void main() {
    float r = length(vLocal);
    if (r > 1.0 || vIntensity <= 0.0) {
        discard;
    }

    // Bright plateau over the built-up area, then a long soft skirt that is
    // windowed to zero at the sprite edge.
    float u = r * HaloScale;
    float glow = pow(1.0 / (1.0 + u * u), 1.2) * (1.0 - smoothstep(0.7, 1.0, r));
    float intensity = clamp(glow * vIntensity * 0.8, 0.0, 1.0);

    vec3 halo = vec3(1.00, 0.56, 0.24);
    vec3 core = vec3(1.00, 0.92, 0.72);
    vec3 color = mix(halo, core, smoothstep(0.15, 0.8, glow));
    color = mix(color, vec3(0.88, 0.93, 1.00), step(0.8, vSeed) * 0.3);
    outColor = vec4(color, intensity);
}
