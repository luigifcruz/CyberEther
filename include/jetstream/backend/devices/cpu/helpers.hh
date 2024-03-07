#ifndef JETSTREAM_BACKEND_DEVICE_CPU_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_CPU_HELPERS_HH

#include <math.h>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream::Backend {

// When strict Atan2 is not necessary.
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
                return ApproxAtan(z) + JST_PI;
            } else {
                // atan2(y,x) = atan(y/x) - PI if x < 0, y < 0
                return ApproxAtan(z) - JST_PI;
            }
        } else {
            // Use property atan(y/x) = PI/2 - atan(x/y) if |y/x| > 1.
            const F32 z = x / y;
            if (y > 0.0) {
                // atan2(y,x) = PI/2 - atan(x/y) if |y/x| > 1, y > 0
                return -ApproxAtan(z) + JST_PI_2;
            } else {
                // atan2(y,x) = -PI/2 - atan(x/y) if |y/x| > 1, y < 0
                return -ApproxAtan(z) - JST_PI_2;
            }
        }
    } else {
        if (y > 0.0f) {
            // x = 0, y > 0
            return JST_PI_2;
        } else if (y < 0.0f) {
            // x = 0, y < 0
            return -JST_PI_2;
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

inline I32 GetSocketBufferSize() {
    I32 bufferSize = 0;
    I32 recommendedBufferSize = 32*1024*1024;  // 32 MB

    (void)recommendedBufferSize;

#ifdef JST_OS_LINUX
    // Get default socket buffer size (rmem_default).

    I32 defaultSocketSize = 0;
    {
        std::ifstream file("/proc/sys/net/core/rmem_default");
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            defaultSocketSize = std::stoi(line);
        }
    }

    if (defaultSocketSize < recommendedBufferSize) {
        bufferSize = recommendedBufferSize;
        JST_INFO("Increasing socket buffer size to {:.2f} MB.", static_cast<F32>(bufferSize) / JST_MB);
    }

    // Get maximum socket buffer size (rmem_max).

    I32 maxSocketSize = 0;
    {
        std::ifstream file("/proc/sys/net/core/rmem_max");
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            maxSocketSize = std::stoi(line);
        }
    }

    if (maxSocketSize < bufferSize) {
        bufferSize = maxSocketSize;
        JST_WARN("Can't increase socket buffer size to {:.2f} MB. Maximum system socket buffer size is {:.2f} MB.", 
                 static_cast<F32>(bufferSize) / JST_MB, static_cast<F32>(maxSocketSize) / JST_MB);
    }
#endif

#ifdef JST_OS_MAC
    bufferSize = recommendedBufferSize;
    JST_INFO("Setting socket buffer size to {:.2f} MB.", static_cast<F32>(bufferSize) / JST_MB);
#endif

    JST_DEBUG("Socket buffer size: {:.2f} MB.", static_cast<F32>(bufferSize) / JST_MB);

    return bufferSize;
}

inline std::pair<F32, F32> ComputePerpendicular(const std::pair<F32, F32>& d, const std::pair<F32, F32>& thickness) {
    const auto& [tx, ty] = thickness;
    auto [dx, dy] = d;

    // Compute length
    const F32& len = std::sqrt(dx * dx + dy * dy);

    // Normalize
    dx /= len;
    dy /= len;

    // Compute perpendicular (normalized)
    return {-dy * tx, dx * ty};
}

inline Result GenerateThickenedLine(Tensor<Device::CPU, F32>& thick, 
                                    const Tensor<Device::CPU, F32>& line, 
                                    const std::pair<F32, F32>& thickness) {
    const U64 numberOfElements = line.size() / 3;

    for (U64 i = 0; i < numberOfElements - 1; ++i) {
        const F32 x1 = line[i * 3];
        const F32 y1 = line[i * 3 + 1];
        const F32 x2 = line[(i + 1) * 3];
        const F32 y2 = line[(i + 1) * 3 + 1];

        const auto& [perpX, perpY] = ComputePerpendicular({x2 - x1, y2 - y1}, thickness);

        // Index for the newPlot vector
        const U64 idx = i * 4 * 3;

        // Upper left
        thick[idx] = x1 + perpX;
        thick[idx + 1] = y1 + perpY;
        thick[idx + 2] = 0.0f;

        // Lower left
        thick[idx + 3] = x1 - perpX;
        thick[idx + 4] = y1 - perpY;
        thick[idx + 5] = 0.0f;

        // Upper right
        thick[idx + 6] = x2 + perpX;
        thick[idx + 7] = y2 + perpY;
        thick[idx + 8] = 0.0f;

        // Lower right
        thick[idx + 9] = x2 - perpX;
        thick[idx + 10] = y2 - perpY;
        thick[idx + 11] = 0.0f;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Backend

#endif