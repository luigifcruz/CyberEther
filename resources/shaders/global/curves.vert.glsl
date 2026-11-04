#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 frame;
    vec4 params;
} uniforms;

layout(location = 0) in vec2 inVertex;
layout(location = 1) in vec4 inStart;
layout(location = 2) in vec4 inEnd;
layout(location = 3) in vec4 inColor;
layout(location = 4) in vec4 inStroke;

layout(location = 0) out vec4 outColor;

vec2 bezier(float index, float segments) {
    float t = index / segments;
    float u = 1.0 - t;
    vec2 p = u * u * u * inStart.xy +
             3.0 * u * u * t * inStart.zw +
             3.0 * u * t * t * inEnd.xy +
             t * t * t * inEnd.zw;
    return uniforms.frame.zw + p * uniforms.params.x;
}

vec2 segmentNormal(vec2 a, vec2 b) {
    vec2 d = b - a;
    float d2 = dot(d, d);
    if (d2 > 0.0) {
        d *= inversesqrt(d2);
    }
    return vec2(d.y, -d.x);
}

void main() {
    float segments = clamp(inStroke.y, 1.0, uniforms.params.z);
    float index = min(inVertex.x, segments);
    float side = inVertex.y;

    vec2 point = bezier(index, segments);
    vec2 normal;
    if (index < 0.5) {
        normal = segmentNormal(point, bezier(1.0, segments));
    } else if (index > segments - 0.5) {
        normal = segmentNormal(bezier(segments - 1.0, segments), point);
    } else {
        vec2 before = segmentNormal(bezier(index - 1.0, segments), point);
        vec2 after = segmentNormal(point, bezier(index + 1.0, segments));
        normal = (before + after) * 0.5;
        float d2 = dot(normal, normal);
        if (d2 > 0.000001) {
            normal *= min(1.0 / d2, 100.0);
        }
    }

    float fringe = uniforms.params.y;
    float thickness = max(inStroke.x, 1.0);
    float inner = inStroke.x > fringe ? (thickness - fringe) * 0.5 : 0.0;
    float offset = abs(side) > 1.5 ? inner + fringe : inner;
    vec2 position = point + normal * offset * sign(side);

    vec2 size = max(uniforms.frame.xy, vec2(1.0));
    gl_Position = vec4(position.x / size.x * 2.0 - 1.0,
                       1.0 - position.y / size.y * 2.0,
                       1.0,
                       1.0);
    outColor = abs(side) > 1.5 ? vec4(inColor.rgb, 0.0) : inColor;
}
