#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inLocal;
layout(location = 1) in float inHeading;

layout(location = 0) out vec4 outColor;

vec2 planeIconPoint(float x, float y, float scale) {
    vec2 v = vec2((x - 12.0) * scale, (12.0 - y) * scale);
    return vec2(-v.y, v.x);
}

float sdPolygon20(vec2 p, vec2 verts[20]) {
    float d = dot(p - verts[0], p - verts[0]);
    float s = 1.0;

    for (int i = 0, j = 19; i < 20; j = i, ++i) {
        vec2 e = verts[j] - verts[i];
        vec2 w = p - verts[i];
        vec2 b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(b, b));

        bvec3 c = bvec3(p.y >= verts[i].y,
                        p.y < verts[j].y,
                        e.x * w.y > e.y * w.x);
        if (all(c) || all(not(c))) {
            s = -s;
        }
    }

    return s * sqrt(d);
}

void main() {
    vec2 forward = vec2(sin(inHeading), cos(inHeading));
    vec2 right = vec2(forward.y, -forward.x);
    vec2 local = vec2(dot(inLocal, right), dot(inLocal, forward));

    const vec3 markerColor = vec3(1.0, 0.3, 0.2);

    const float iconScale = 1.9;
    vec2 poly[20] = vec2[](
        planeIconPoint(16.0, 10.0, iconScale),
        planeIconPoint(20.0, 10.0, iconScale),
        planeIconPoint(21.414, 10.586, iconScale),
        planeIconPoint(22.0, 12.0, iconScale),
        planeIconPoint(21.414, 13.414, iconScale),
        planeIconPoint(20.0, 14.0, iconScale),
        planeIconPoint(16.0, 14.0, iconScale),
        planeIconPoint(12.0, 21.0, iconScale),
        planeIconPoint(9.0, 21.0, iconScale),
        planeIconPoint(11.0, 14.0, iconScale),
        planeIconPoint(7.0, 14.0, iconScale),
        planeIconPoint(5.0, 16.0, iconScale),
        planeIconPoint(2.0, 16.0, iconScale),
        planeIconPoint(4.0, 12.0, iconScale),
        planeIconPoint(2.0, 8.0, iconScale),
        planeIconPoint(5.0, 8.0, iconScale),
        planeIconPoint(7.0, 10.0, iconScale),
        planeIconPoint(11.0, 10.0, iconScale),
        planeIconPoint(9.0, 3.0, iconScale),
        planeIconPoint(12.0, 3.0, iconScale)
    );

    float planeSdf = sdPolygon20(local, poly);
    const float aa = 1.0;
    float alpha = 1.0 - smoothstep(0.0, aa, planeSdf);

    if (alpha < 0.01) {
        discard;
    }

    outColor = vec4(markerColor, alpha);
}
