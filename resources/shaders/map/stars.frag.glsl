#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vLocal;
layout(location = 1) in float vBrightness;
layout(location = 2) in float vSeed;

layout(location = 0) out vec4 outColor;

// Map a 0..1 hash to a blackbody-ish star color: mostly blue-white/white,
// with a tail of yellow, orange, and red giants.
vec3 starColor(float h) {
    if (h < 0.55) {
        return mix(vec3(0.75, 0.82, 1.00), vec3(1.00, 1.00, 1.00), h / 0.55);
    } else if (h < 0.80) {
        return mix(vec3(1.00, 1.00, 1.00), vec3(1.00, 0.92, 0.74),
                   (h - 0.55) / 0.25);
    } else if (h < 0.93) {
        return mix(vec3(1.00, 0.92, 0.74), vec3(1.00, 0.78, 0.52),
                   (h - 0.80) / 0.13);
    }
    return mix(vec3(1.00, 0.78, 0.52), vec3(1.00, 0.62, 0.42),
               (h - 0.93) / 0.07);
}

void main() {
    float r = length(vLocal);
    if (r > 1.0) discard;

    // Soft core + faint halo.
    float core = exp(-r * r * 6.0);
    float halo = exp(-r * 4.0) * 0.25;
    float glow = core + halo;

    // Diffraction spikes on the brightest stars.
    float spike = 0.0;
    if (vBrightness > 0.6) {
        float ax = min(abs(vLocal.x), abs(vLocal.y));
        spike = exp(-ax * 30.0) * exp(-r * 1.5) *
                smoothstep(0.6, 1.0, vBrightness) * 0.5;
    }

    float intensity = clamp((glow + spike) * vBrightness, 0.0, 1.0);
    vec3 col = starColor(vSeed);

    // Premultiplied-style: rgb is full color, alpha scales it over black.
    outColor = vec4(col, intensity);
}
