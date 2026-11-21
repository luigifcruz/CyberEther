#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 rect;
    vec4 grid;
    vec4 params;
    vec4 listeningColor;
    vec4 busyColor;
    vec4 dimColor;
    vec4 shape;
} uniforms;

layout(location = 0) in vec2 inUv;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265;

float trace(float x, float time, float amp, float freq, float speed, float phase) {
    return amp * sin(2.0 * PI * freq * x - time * speed + phase);
}

float beam(float offset) {
    return exp(-offset * offset * 1.8) + 0.22 * exp(-offset * offset * 0.22);
}

void main() {
    vec2 size = uniforms.grid.xy;
    float pitch = uniforms.grid.z;
    float radius = uniforms.grid.w;
    float time = uniforms.params.x;
    float busy = clamp(uniforms.params.y, 0.0, 1.0);
    float activity = clamp(uniforms.params.z, 0.0, 1.0);
    float level = clamp(uniforms.params.w, 0.0, 1.0);

    vec2 counts = max(floor(size / pitch), vec2(1.0));
    vec2 origin = (size - counts * pitch) * 0.5;
    vec2 px = inUv * size;
    vec2 cell = floor((px - origin) / pitch);
    if (any(lessThan(cell, vec2(0.0))) || any(greaterThanEqual(cell, counts))) {
        discard;
    }
    vec2 center = origin + (cell + 0.5) * pitch;
    float corner = min(uniforms.shape.x, min(size.x, size.y) * 0.5);
    vec2 q = abs(center - size * 0.5) - (size * 0.5 - corner);
    float edge = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - corner;
    if (edge > -uniforms.shape.y) {
        discard;
    }
    float x = center.x / size.x;
    float y = (center.y / size.y - 0.5) * 2.0;
    float rows = counts.y * 0.5;

    float taper = pow(max(sin(PI * x), 0.0), 0.8);
    float fade = smoothstep(0.0, 0.06, x) * (1.0 - smoothstep(0.94, 1.0, x));
    float amp = (0.1 * activity + 1.25 * level) * taper;
    float sway = 0.65 + 0.35 * sin(time * 0.7);

    float a = trace(x, time, amp * sway, 1.25, 2.3, 0.0) + trace(x, time, amp * 0.35, 3.3, 3.9, 0.6);
    float b = trace(x, time, amp * 0.75, 2.05, -2.9, 1.7) - trace(x, time, amp * 0.25, 3.3, 3.9, 0.6);

    float lit = max(beam(abs(y - a) * rows), beam(abs(y - b) * rows) * smoothstep(0.02, 0.25, level));
    float shimmer = 0.62 + 0.38 * sin(x * 13.0 - time * 4.5);
    lit *= mix(shimmer, 1.0, smoothstep(0.35, 1.0, activity));
    lit *= mix(0.35, 1.0, activity) * fade;

    float glow = clamp(lit, 0.0, 1.0);
    float base = 0.16 * fade;
    float energy = base + glow * (1.0 - base);

    vec3 tint = mix(uniforms.listeningColor.rgb, uniforms.busyColor.rgb, busy);
    vec3 color = mix(uniforms.dimColor.rgb, tint, smoothstep(0.08, 0.55, glow));
    color = mix(color, vec3(1.0), smoothstep(0.55, 1.0, glow) * 0.7);

    float spot = radius * (0.8 + 0.35 * glow);
    float reach = length(px - center);
    float mask = 1.0 - smoothstep(spot - 0.75, spot + 0.75, reach);
    outColor = vec4(color, mask * energy);
}
