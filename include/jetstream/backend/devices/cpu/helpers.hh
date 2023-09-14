#ifndef JETSTREAM_BACKEND_DEVICE_CPU_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_CPU_HELPERS_HH

#include <math.h>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Backend {

// When strict Atan2 is not necessary. YOLO.
// Adapted from https://www.dsprelated.com/showarticle/1052.php.
inline F32 ApproxAtan(const F32& z) {
    const F32 n1 =  0.97239411f;
    const F32 n2 = -0.19194795f;
    return (n1 + n2 * z * z) * z;
}

inline F32 ApproxAtan2(const F32& y, const F32& x) {
    if (x != 0.0f) {
        if (fabsf(x) > fabsf(y)) {
            const F32 z = y / x;
            if (x > 0.0) {
                // atan2(y,x) = atan(y/x) if x > 0
                return ApproxAtan(z);
            } else if (y >= 0.0) {
                // atan2(y,x) = atan(y/x) + PI if x < 0, y >= 0
                return ApproxAtan(z) + M_PI;
            } else {
                // atan2(y,x) = atan(y/x) - PI if x < 0, y < 0
                return ApproxAtan(z) - M_PI;
            }
        } else {
            // Use property atan(y/x) = PI/2 - atan(x/y) if |y/x| > 1.
            const F32 z = x / y;
            if (y > 0.0) {
                // atan2(y,x) = PI/2 - atan(x/y) if |y/x| > 1, y > 0
                return -ApproxAtan(z) + M_PI_2;
            } else {
                // atan2(y,x) = -PI/2 - atan(x/y) if |y/x| > 1, y < 0
                return -ApproxAtan(z) - M_PI_2;
            }
        }
    } else {
        if (y > 0.0f) {
            // x = 0, y > 0
            return M_PI_2;
        } else if (y < 0.0f) {
            // x = 0, y < 0
            return -M_PI_2;
        }
    }
    return 0.0f; // x,y = 0. Could return NaN instead.
}

// When strict Log10 is not necessary. YOLO.
// Adapted from http://openaudio.blogspot.com/2017/02/faster-log10-and-pow.html
inline F32 ApproxLog10(const F32& X) {
    F32 Y, F;
    int E;
    F = frexpf(fabs(X), &E);
    Y = 1.23149591368684f;
    Y *= F;
    Y += -4.11852516267426f;
    Y *= F;
    Y += 6.02197014179219f;
    Y *= F;
    Y += -3.13396450166353f;
    Y += E;
    return Y * 0.3010299956639812f;
}

}  // namespace Jetstream::Backend

#endif