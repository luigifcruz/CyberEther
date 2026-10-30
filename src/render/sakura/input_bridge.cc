#include <jetstream/render/sakura/clipboard.hh>

#include <jetstream/render/base/window.hh>

#include "context.hh"
#include "input.hh"

namespace Jetstream::Sakura {

void SetClipboardText(Render::Window* window, const std::string& value) {
    if (window) {
        window->setClipboardText(value);
    }
}

std::string GetClipboardText(Render::Window* window) {
    return window ? window->clipboardText() : std::string();
}

namespace Private {

void SampleInput(InputFrame& frame, const Render::InputState& state) {
    if (frame.source == &state && frame.sourceGeneration == state.generation() &&
        frame.sourceFrame == static_cast<I64>(state.frame())) {
        return;
    }
    frame.source = &state;
    frame.sourceGeneration = state.generation();
    frame.sourceFrame = static_cast<I64>(state.frame());

    frame.advance(state.deltaTime());
    frame.time = state.time();
    frame.windowFocused = state.windowFocused();
    frame.modifiers = state.modifiers();
    frame.pointer = state.pointer();
    frame.buttons = state.buttons();

    frame.events.insert(frame.events.end(), state.events().begin(), state.events().end());
}

void ResetInput(InputFrame& frame) {
    frame.advance(0.0f);
    frame.source = nullptr;
    frame.sourceGeneration = 0;
    frame.sourceFrame = -1;
    frame.windowFocused = false;
    frame.pointer.reset();
    frame.buttons.fill(false);
    frame.modifiers = {};
}

void ApplyCursor(const Context& ctx, SurfaceCursor cursor) {
    if (cursor != SurfaceCursor::Default) {
        ctx.render->requestCursor(cursor);
    }
}

}  // namespace Private

}  // namespace Jetstream::Sakura
