#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inAircraft;

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

layout(location = 0) out vec2 outLocal;
layout(location = 1) out float outHeading;

const float PI = 3.14159265358979323846;
const float MAX_MERCATOR_LAT = 85.05112878;
const float MARKER_RADIUS_PX = 26.0;

float mercatorX(float lon) {
    return (lon + 180.0) / 360.0;
}

float mercatorY(float lat) {
    lat = clamp(lat, -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT);
    float r = radians(lat);
    return (1.0 - log(tan(r) + 1.0 / cos(r)) / PI) / 2.0;
}

void main() {
    float acLat = inAircraft.x;
    float acLon = inAircraft.y;

    float cx = mercatorX(uniforms.centerLon);
    float cy = mercatorY(uniforms.centerLat);
    float scale = pow(2.0, uniforms.zoom);

    float acMx = mercatorX(acLon);
    float acMy = mercatorY(acLat);

    float acNdcX = (acMx - cx) * scale * 2.0 / uniforms.aspectRatio;
    float acNdcY = (cy - acMy) * scale * 2.0;

    float safeViewWidth = float(max(uniforms.viewWidth, 1));
    float safeViewHeight = float(max(uniforms.viewHeight, 1));

    outLocal = inPosition.xy * MARKER_RADIUS_PX;
    vec2 offsetNdc = vec2((outLocal.x * 2.0 * uniforms.surfaceScale) / safeViewWidth,
                          (outLocal.y * 2.0 * uniforms.surfaceScale) / safeViewHeight);

    gl_Position = vec4(acNdcX + offsetNdc.x, acNdcY + offsetNdc.y, 0.0, 1.0);
    outHeading = radians(inAircraft.z);
}
