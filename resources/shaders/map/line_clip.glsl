// Clip a minor great-circle arc, not just its endpoints, to the visible cap.
// A long border may cross the viewport with BOTH endpoints below the horizon.
// Shared with the CPU dash layout and its tests via geomap_lines.hh.
#ifdef __cplusplus
inline bool clipLineToHorizon(vec3& a, vec3& b, vec3 n, float threshold) {
#else
bool clipLineToHorizon(inout vec3 a, inout vec3 b, vec3 n, float threshold) {
#endif
    if (dot(a, n) >= threshold && dot(b, n) >= threshold) return true;

    vec3 sum = a + b;
    float sumLength = length(sum);
    // Conservative arc bound: distance from the midpoint to any arc point
    // is at most the full chord length. Reject distant geometry before trig.
    if (sumLength < 1e-7 ||
        dot(sum, n) / sumLength + length(b - a) < threshold) return false;

    vec3 plane = cross(a, b);
    float planeLength = length(plane);
    if (planeLength < 1e-7) return false;
    plane /= planeLength;

    vec3 closest = n - plane * dot(n, plane);
    float peak = length(closest);
    if (peak < threshold) return false;
    closest /= peak;
    vec3 tangent = cross(plane, closest);

    const float tau = 6.283185307179586;
    float begin = atan(dot(a, tangent), dot(a, closest));
    float end = atan(dot(b, tangent), dot(b, closest));
    if (end < begin) end += tau;
    float limit = acos(clamp(threshold / peak, 0.0f, 1.0f));
    float center = floor((begin + end) * 0.5 / tau + 0.5) * tau;
    float clippedBegin = max(begin, center - limit);
    float clippedEnd = min(end, center + limit);
    if (clippedEnd <= clippedBegin) return false;

    if (clippedBegin > begin) {
        a = closest * cos(clippedBegin) + tangent * sin(clippedBegin);
    }
    if (clippedEnd < end) {
        b = closest * cos(clippedEnd) + tangent * sin(clippedEnd);
    }
    return true;
}
