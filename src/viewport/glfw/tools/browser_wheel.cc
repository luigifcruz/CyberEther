#include "browser_wheel.hh"

#include <emscripten/html5.h>
#include <emscripten/proxying.h>
#include <pthread.h>

#include <utility>

namespace Jetstream::Viewport {

struct BrowserWheel::State {
    Classifier classifier;
    bool pending = false;
    bool precise = false;

    static EM_BOOL OnWheel(int, const EmscriptenWheelEvent* event, void* user) {
        static_cast<State*>(user)->wheel(*event);
        return EM_FALSE;
    }

    static void Release(void* user) {
        delete static_cast<State*>(user);
    }

    void wheel(const EmscriptenWheelEvent& event) {
        precise = classifier.classify({
            .time = event.mouse.timestamp,
            .pixels = event.deltaMode == DOM_DELTA_PIXEL,
            .pinch = static_cast<bool>(event.mouse.ctrlKey),
            .deltaX = event.deltaX,
            .deltaY = event.deltaY,
        });
        pending = true;
    }
};

BrowserWheel::BrowserWheel() : state(std::make_unique<State>()) {
    emscripten_set_wheel_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, state.get(), EM_TRUE, &State::OnWheel);
}

BrowserWheel::~BrowserWheel() {
    emscripten_set_wheel_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, EM_TRUE, nullptr);
    State* retired = state.release();
    if (!emscripten_proxy_async(emscripten_proxy_get_system_queue(), pthread_self(), &State::Release, retired)) {
        State::Release(retired);
    }
}

bool BrowserWheel::take(Extent2D<F32>& delta) {
    if (!std::exchange(state->pending, false) || !state->precise) {
        return false;
    }
    delta = Classifier::Normalize(delta);
    return true;
}

}  // namespace Jetstream::Viewport
