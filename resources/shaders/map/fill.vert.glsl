#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    float centerLon;
    float centerLat;
    float zoom;
    float aspectRatio;
    float surfaceScale;
    float lineWidth;
    float colorR;
    float colorG;
    float colorB;
    float viewportWidth;
    float viewportHeight;
} uniforms;

// Per-vertex: (lon, lat, r, g, b).
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vColor;

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

void main() {
    float cx = mercatorX(uniforms.centerLon);
    float cy = mercatorY(uniforms.centerLat);
    float scale = pow(2.0, uniforms.zoom);

    float vx = (mercatorX(inPosition.x) - cx) * scale * 2.0;
    float vy = (cy - mercatorY(inPosition.y)) * scale * 2.0;
    vx /= uniforms.aspectRatio;

    gl_Position = vec4(vx, vy, 0.0, 1.0);
    vColor = inColor;
}
