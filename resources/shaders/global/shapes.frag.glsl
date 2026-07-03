#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inLocalPos;
layout(location = 1) in vec4 inFillColor;
layout(location = 2) in vec4 inBorderColor;
layout(location = 3) in vec4 inShapeParams; // x=type, y=borderWidth, z=cornerRadius, w=unused
layout(location = 4) in vec2 inPixelSize;

layout(location = 0) out vec4 outColor;

// Shape type constants
const int TYPE_TRIANGLE = 0;
const int TYPE_RECT = 1;
const int TYPE_CIRCLE = 2;

float sdfCircle(vec2 p, float radius) {
    return length(p) - radius;
}

float sdfTriangle(vec2 p) {
    // Optimized equilateral triangle for better antialiasing
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 0.5;
    p.y = p.y + 1.0 / sqrt(3.0);
    if (p.x + k * p.y > 0.0) {
        p = vec2(p.x - k * p.y, -k * p.x - p.y) * 0.5;
    }
    p.x -= clamp(p.x, -2.0 / sqrt(3.0), 2.0 / sqrt(3.0));
    return -length(p) * sign(p.y);
}

float alphaFromSignedDistance(float signedDistance) {
    // Convert local SDF distance to stable one-pixel screen coverage.
    float distancePerPixel = length(vec2(dFdx(signedDistance), dFdy(signedDistance)));
    distancePerPixel = max(distancePerPixel, 1.0e-6);
    return clamp(0.5 - signedDistance / distancePerPixel, 0.0, 1.0);
}

float alphaRect(vec2 p, vec2 size) {
    // Use separable box coverage to keep thin rectangles crisp.
    vec2 halfSize = size * 0.5;
    vec2 insideDistance = halfSize - abs(p);

    vec2 localPixel = max(fwidth(p), vec2(1.0e-6));

    vec2 coverage = clamp(insideDistance / localPixel + 0.5, 0.0, 1.0);
    return coverage.x * coverage.y;
}

float alphaRoundedRect(vec2 p, vec2 size, float radiusPixels) {
    vec2 localPixel = max(fwidth(p), vec2(1.0e-6));
    vec2 halfSizePixels = (size * 0.5) / localPixel;
    float radius = clamp(radiusPixels, 0.0, min(halfSizePixels.x, halfSizePixels.y));

    vec2 pPixels = p / localPixel;
    vec2 q = abs(pPixels) - (halfSizePixels - vec2(radius));
    float signedDistance = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    return clamp(0.5 - signedDistance, 0.0, 1.0);
}

float roundedRectSdPixels(vec2 p, float radiusPixels) {
    vec2 localPixel = max(fwidth(p), vec2(1.0e-6));
    vec2 halfSizePixels = (vec2(0.5)) / localPixel;
    float radius = clamp(radiusPixels, 0.0, min(halfSizePixels.x, halfSizePixels.y));

    vec2 pPixels = p / localPixel;
    vec2 q = abs(pPixels) - (halfSizePixels - vec2(radius));
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

float shapeSdPixels(float localSignedDistance) {
    float distancePerPixel = max(length(vec2(dFdx(localSignedDistance), dFdy(localSignedDistance))), 1.0e-6);
    return localSignedDistance / distancePerPixel;
}

void main() {
    int shapeType = int(inShapeParams.x);
    float borderWidth = inShapeParams.y;
    float cornerRadius = inShapeParams.z;

    if (borderWidth <= 0.0) {
        float fillAlpha = 0.0;
        if (shapeType == TYPE_CIRCLE) {
            fillAlpha = alphaFromSignedDistance(sdfCircle(inLocalPos, 0.5));
        } else if (shapeType == TYPE_RECT) {
            fillAlpha = cornerRadius > 0.0 ? alphaRoundedRect(inLocalPos, vec2(1.0), cornerRadius)
                                           : alphaRect(inLocalPos, vec2(1.0));
        } else if (shapeType == TYPE_TRIANGLE) {
            fillAlpha = alphaFromSignedDistance(sdfTriangle(inLocalPos));
        }
        outColor = vec4(inFillColor.rgb, inFillColor.a * fillAlpha);
        return;
    }

    float sdPixels;
    if (shapeType == TYPE_RECT) {
        sdPixels = roundedRectSdPixels(inLocalPos, cornerRadius);
    } else if (shapeType == TYPE_CIRCLE) {
        sdPixels = shapeSdPixels(sdfCircle(inLocalPos, 0.5));
    } else { // TYPE_TRIANGLE
        sdPixels = shapeSdPixels(sdfTriangle(inLocalPos));
    }

    float outer = clamp(0.5 - sdPixels, 0.0, 1.0);
    float inner = clamp(0.5 - (sdPixels + borderWidth), 0.0, 1.0);

    vec3 rgb = mix(inBorderColor.rgb, inFillColor.rgb, inner);
    float alpha = outer * mix(inBorderColor.a, inFillColor.a, inner);
    outColor = vec4(rgb, alpha);
}
