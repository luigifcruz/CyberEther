#ifndef JETSTREAM_VIEWPORT_GLFW_BROWSER_WHEEL_HH
#define JETSTREAM_VIEWPORT_GLFW_BROWSER_WHEEL_HH

#include <cmath>
#include <limits>
#include <memory>

#include "jetstream/types.hh"

namespace Jetstream::Viewport {

class BrowserWheel {
 public:
    class Classifier {
     public:
        struct Sample {
            F64 time = 0.0;
            bool pixels = false;
            bool pinch = false;
            F64 deltaX = 0.0;
            F64 deltaY = 0.0;
        };

        bool classify(const Sample& sample) {
            const bool notch = sample.deltaX == 0.0 && std::abs(sample.deltaY) >= kNotchPixels &&
                               sample.deltaY == std::round(sample.deltaY);
            const bool continuing = sample.time - lastPrecise < kGestureWindow;
            const bool precise = sample.pixels && !sample.pinch && (!notch || continuing);
            if (precise) {
                lastPrecise = sample.time;
            }
            return precise;
        }

        static Extent2D<F32> Normalize(const Extent2D<F32>& delta) {
            return {delta.x * kPortPixelsPerUnit / kPixelsPerUnit, delta.y * kPortPixelsPerUnit / kPixelsPerUnit};
        }

     private:
        static constexpr F64 kNotchPixels = 50.0;
        static constexpr F64 kGestureWindow = 100.0;
        static constexpr F32 kPixelsPerUnit = 10.0f;
        static constexpr F32 kPortPixelsPerUnit = 100.0f;

        F64 lastPrecise = -std::numeric_limits<F64>::infinity();
    };

    BrowserWheel();
    ~BrowserWheel();

    BrowserWheel(const BrowserWheel&) = delete;
    BrowserWheel& operator=(const BrowserWheel&) = delete;

    bool take(Extent2D<F32>& delta);

 private:
    struct State;

    std::unique_ptr<State> state;
};

}  // namespace Jetstream::Viewport

#endif
