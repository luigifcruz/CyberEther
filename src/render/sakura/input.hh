#ifndef JETSTREAM_RENDER_SAKURA_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_INPUT_HH

#include <jetstream/render/base/input.hh>
#include <jetstream/surface.hh>
#include <jetstream/types.hh>

#include <array>
#include <optional>
#include <vector>

namespace Jetstream::Sakura {

using FrameEvent = Render::InputState::Event;

struct InputFrame {
    U64 serial = 0;
    F64 time = 0.0;
    F32 deltaTime = 0.0f;
    bool windowFocused = true;
    KeyModifiers modifiers;
    std::optional<Extent2D<F32>> pointer;
    std::array<bool, 3> buttons{};
    std::vector<FrameEvent> events;
    const void* source = nullptr;
    U64 sourceGeneration = 0;
    I64 sourceFrame = -1;

    void advance(F32 delta) {
        ++serial;
        time += delta;
        deltaTime = delta;
        events.clear();
    }
};

struct Context;

namespace Private {

void SampleInput(InputFrame& frame, const Render::InputState& state);
void ResetInput(InputFrame& frame);
void ApplyCursor(const Context& ctx, SurfaceCursor cursor);

}  // namespace Private

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_INPUT_HH
