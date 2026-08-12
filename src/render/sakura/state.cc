#include "state.hh"

namespace Jetstream::Sakura::Private {

namespace {

bool& KeyboardInputCaptured() {
    static bool captured = false;
    return captured;
}

}  // namespace

void SetKeyboardInputCaptured(const bool captured) {
    KeyboardInputCaptured() = captured;
}

bool IsKeyboardInputCaptured() {
    return KeyboardInputCaptured();
}

}  // namespace Jetstream::Sakura::Private
