#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    float centerLon;
    float centerLat;
    float zoom;
    float aspectRatio;
    float surfaceScale;
    int viewWidth;
    int viewHeight;
    int aircraftCount;
    int _pad0;
} uniforms;

layout(set = 0, binding = 1) readonly buffer AircraftBuffer {
    vec4 aircraft[];  // lat, lon, heading, altitude
};

const float PI = 3.14159265358979323846;
const float MAX_MERCATOR_LAT = 85.05112878;

float mercatorX(float lon) {
    return (lon + 180.0) / 360.0;
}

float mercatorY(float lat) {
    lat = clamp(lat, -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT);
    float r = radians(lat);
    return (1.0 - log(tan(r) + 1.0 / cos(r)) / PI) / 2.0;
}

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
    // Convert fragment position to NDC [-1, 1].
    vec2 fragNdc = inTexcoord * 2.0 - 1.0;

    // Project center through Mercator.
    float cx = mercatorX(uniforms.centerLon);
    float cy = mercatorY(uniforms.centerLat);

    float scale = pow(2.0, uniforms.zoom);

    const float markerRadiusPx = 26.0;

    const vec3 markerColor = vec3(1.0, 0.3, 0.2);
    float coverage = 0.0;

    for (int i = 0; i < uniforms.aircraftCount && i < 256; ++i) {
        float acLat = aircraft[i].x;
        float acLon = aircraft[i].y;
        float acHeading = aircraft[i].z;

        // Project aircraft position through Mercator.
        float acMx = mercatorX(acLon);
        float acMy = mercatorY(acLat);

        // To NDC (same as vertex shader).
        float acNdcX = (acMx - cx) * scale * 2.0 / uniforms.aspectRatio;
        float acNdcY = (cy - acMy) * scale * 2.0;

        // Distance in pixels.
        float dx = (fragNdc.x - acNdcX) * uniforms.viewWidth * 0.5 / uniforms.surfaceScale;
        float dy = (fragNdc.y - acNdcY) * uniforms.viewHeight * 0.5 / uniforms.surfaceScale;

        // Fast reject outside the marker's outer radius.
        float distSq = dx * dx + dy * dy;
        if (distSq > markerRadiusPx * markerRadiusPx) {
            continue;
        }

        // Convert heading (degrees clockwise from North) to screen axes.
        float headingRad = radians(acHeading);
        vec2 forward = vec2(sin(headingRad), cos(headingRad));
        vec2 right = vec2(forward.y, -forward.x);
        vec2 local = vec2(dot(vec2(dx, dy), right), dot(vec2(dx, dy), forward));

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

        if (alpha > 0.001) {
            coverage = alpha + coverage * (1.0 - alpha);
            if (coverage > 0.995) {
                break;
            }
        }
    }

    if (coverage < 0.01) {
        discard;
    }

    // Output straight alpha for the pipeline's SRC_ALPHA blending mode.
    outColor = vec4(markerColor, coverage);
}
