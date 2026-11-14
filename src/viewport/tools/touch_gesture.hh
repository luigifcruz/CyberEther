#ifndef JETSTREAM_VIEWPORT_TOOLS_TOUCH_GESTURE_HH
#define JETSTREAM_VIEWPORT_TOOLS_TOUCH_GESTURE_HH

#include <vector>

#include "jetstream/render/base/input.hh"
#include "scroll_momentum.hh"

namespace Jetstream::Viewport {

class TouchGesture {
 public:
    struct Event {
        enum class Type { Down, Motion, Up, Frame, Cancel };
        Type type;
        I64 id = 0;
        Extent2D<F32> position = {0.0f, 0.0f};
        U32 timestamp = 0;
        bool momentumAllowed = true;
    };

    explicit TouchGesture(Render::InputQueue& input);
    TouchGesture(const TouchGesture&) = delete;
    TouchGesture& operator=(const TouchGesture&) = delete;

    void handle(const Event& event, F64 now, bool allowStart = true);
    void poll(F64 now);

    bool active() const;
    bool ownsPointer() const { return pointerOwned; }
    void resumePointer();

 private:
    enum class Action { Pending, Swipe, ContextMenu, Pinch, Ignored };
    struct Contact {
        I64 id;
        Extent2D<F32> position;
    };

    Render::InputQueue& input;
    std::vector<Contact> contacts;
    Action action = Action::Pending;
    bool pointerOwned = false;
    bool momentumAllowed = true;
    bool coasting = false;
    F64 steppedAt = 0.0;
    ScrollMomentum momentum;
    Extent2D<F32> origin = {0.0f, 0.0f};
    Extent2D<F32> last = {0.0f, 0.0f};
    F64 startedAt = 0.0;
    U32 startedTimestamp = 0;
    bool pinchActive = false;
    bool pinchDirty = false;
    F32 pinchDistance = 0.0f;
    Extent2D<F32> pinchPosition = {0.0f, 0.0f};

    void scroll(U32 timestamp, ScrollPhase phase, Extent2D<F32> delta = {}, bool cancelled = false);
    void release(U32 timestamp, F64 now);
    void stop();
    void cancel();
    void hold(F64 duration);
    F32 contactDistance() const;
    void pinchFrame();
};

}  // namespace Jetstream::Viewport

#endif
