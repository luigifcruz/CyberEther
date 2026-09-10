#ifndef JETSTREAM_MAP_SUN_GLSL
#define JETSTREAM_MAP_SUN_GLSL

// Keep this std140 layout in sync with SunUniforms in geomap_base.cc.
layout(set = 0, binding = 1) uniform SunUniforms {
    vec4 direction;  // xyz = unit vector toward the sun, w = day/night strength
    vec4 night;      // x = darkening, y = twilight width, z = day lift
    vec4 lights;     // x = city glow strength, y = urban fill strength
} sun;

// Sine of the sun elevation for a point on the unit globe.
float sunElevation(vec3 normal) {
    return dot(normal, sun.direction.xyz);
}

// Fraction of full daylight. Illumination already fades at the geometric
// terminator and is gone by nautical twilight.
float sunDaylight(vec3 normal) {
    float width = max(sun.night.y, 1.0e-3);
    return smoothstep(-width, width * 0.45, sunElevation(normal));
}

// City lights switch on right after sunset and are fully lit once the sun is
// about twelve degrees below the horizon.
float sunLights(vec3 normal) {
    return (1.0 - smoothstep(-0.20, 0.02, sunElevation(normal))) *
           sun.direction.w;
}

#endif
