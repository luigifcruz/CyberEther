#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inLocal;
layout(location = 1) flat in vec4 inColor;
layout(location = 2) flat in vec2 inStyle;
layout(location = 0) out vec4 outColor;

float segmentDistance(vec2 p, vec2 a, vec2 b) {
    vec2 d = b - a;
    return length(p - a - d * clamp(dot(p - a, d) / dot(d, d), 0.0, 1.0));
}

void main() {
    float distance;
    if (inStyle.x < 0.5) {
        // Compact, open directional target. Heading is projected through the
        // globe camera on the CPU, measured clockwise from screen north.
        vec2 forward = vec2(sin(inStyle.y), cos(inStyle.y));
        vec2 right = vec2(forward.y, -forward.x);
        vec2 local = vec2(dot(inLocal, right), dot(inLocal, forward));
        distance = segmentDistance(vec2(abs(local.x), local.y),
                                   vec2(0.0, 0.65), vec2(0.60, -0.45)) - 0.08;
    } else if (inStyle.x < 1.5) {
        distance = max(abs(inLocal.x), abs(inLocal.y)) - 0.75; // history square
    } else {
        // No projected heading: a neutral diamond, not a false north heading.
        distance = abs(abs(inLocal.x) + abs(inLocal.y) - 0.7) * 0.7071 - 0.07;
    }
    float aa = max(fwidth(distance), 0.01);
    float alpha = 1.0 - smoothstep(0.0, aa, distance);
    if (alpha < 0.01) discard;
    outColor = vec4(inColor.rgb, inColor.a * alpha);
}
