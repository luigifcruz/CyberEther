#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "../common/noise.glsl"

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 rect;
    vec4 orb;
    vec4 params;
    vec4 colorA;
    vec4 colorB;
    vec4 colorC;
    vec4 colorD;
} uniforms;

layout(location = 0) in vec2 inUv;

layout(location = 0) out vec4 outColor;

float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++) {
        value += amplitude * valueNoise(p);
        p = p * 2.02 + vec3(11.7, 5.3, 2.9);
        amplitude *= 0.5;
    }
    return value;
}

vec3 tonemap(vec3 c) {
    float peak = max(max(c.r, c.g), c.b);
    return peak > 1.0e-4 ? c * ((1.0 - exp(-peak)) / peak) : c;
}

void main() {
    float time = uniforms.params.x;
    float busy = clamp(uniforms.params.y, 0.0, 1.0);
    float activity = clamp(uniforms.params.z, 0.0, 1.0);
    float level = clamp(uniforms.params.w, 0.0, 1.0);

    vec2 p = (inUv - uniforms.orb.xy) * uniforms.orb.zw;
    float r = length(p);
    float angle = atan(p.y, p.x);

    vec3 core = mix(uniforms.colorA.rgb, uniforms.colorB.rgb, busy);
    vec3 accent = mix(uniforms.colorC.rgb, uniforms.colorD.rgb, busy);
    vec3 deep = core * vec3(0.55, 0.50, 0.75);

    float breathe = 0.56 + 0.03 * sin(time * 1.1) * activity + 0.12 * level;
    float swirl = busy * (0.7 * sin(time * 0.6 + r * 6.0) + time * 0.3) + activity * 0.15 * sin(time * 0.35);
    vec2 q = vec2(cos(angle + swirl), sin(angle + swirl)) * r;

    float slow = fbm(vec3(q * 2.2, time * 0.18));
    float fast = fbm(vec3(q * 4.2 + 7.3, time * (0.4 + 0.8 * busy)));
    float wisp = fbm(vec3(q * 1.4 - 2.5, time * 0.12));

    float erosion = (slow - 0.5) * 0.22 + (fast - 0.5) * 0.10 * (0.5 + busy);
    float edge = breathe + erosion * (0.35 + 0.65 * activity);
    float softness = 0.18 + 0.10 * activity + 0.06 * level;
    float body = 1.0 - smoothstep(edge - softness, edge + softness, r);
    float dense = 1.0 - smoothstep(edge * 0.15, edge * 0.95, r);

    float rr = clamp(r / max(edge, 0.001), 0.0, 1.0);
    float z = sqrt(max(0.0, 1.0 - rr * rr));
    vec3 normal = normalize(vec3(p / max(edge, 0.001), z + 0.0001));
    vec3 lightDir = normalize(vec3(-0.45, -0.65, 0.75));
    float diffuse = clamp(dot(normal, lightDir), 0.0, 1.0);
    float rim = pow(1.0 - z, 3.0);

    float veins = smoothstep(0.45, 0.78, fast) * (0.35 + 0.65 * busy);
    float cloud = smoothstep(0.25, 0.85, slow);
    float wispMask = smoothstep(0.40, 0.75, wisp);

    vec3 color = mix(deep, core, cloud);
    color = mix(color, accent, veins * 0.7);
    color = mix(color, accent * 1.15, wispMask * 0.35 * activity);
    color *= 0.65 + 0.55 * diffuse;
    color += accent * rim * (0.70 + 0.50 * level);
    color += core * dense * 0.40;
    color *= mix(0.70, 1.0, activity);

    float haloWidth = 4.5 - 2.0 * level - 1.0 * busy;
    float haloDist = max(0.0, r - edge * 0.85);
    float halo = exp(-haloDist * haloWidth);
    halo *= 0.35 + 0.30 * busy + 0.45 * level;
    halo *= mix(0.55, 1.0, activity);
    halo *= 0.85 + 0.15 * fbm(vec3(q * 1.6, time * 0.2));
    halo *= 1.0 - smoothstep(0.85, 1.0, length(inUv));
    vec3 haloColor = mix(core, accent, 0.20 + 0.30 * busy);

    vec3 rgb = mix(haloColor, color, body);
    float alpha = clamp(body + (1.0 - body) * halo, 0.0, 1.0);
    rgb = rgb * alpha;
    rgb = tonemap(rgb * 2.4) / max(alpha, 1.0e-3);
    outColor = vec4(rgb, sqrt(alpha));
}
