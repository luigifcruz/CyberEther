#ifndef JETSTREAM_RENDER_SAKURA_STATE_HH
#define JETSTREAM_RENDER_SAKURA_STATE_HH

#include "jetstream/types.hh"

namespace Jetstream::Sakura::Private {

void SetKeyboardInputCaptured(bool captured);
bool IsKeyboardInputCaptured();

inline bool ConsumeRequest(U64& seen, U64 request) {
    if (request == seen) {
        return false;
    }
    seen = request;
    return true;
}

}  // namespace Jetstream::Sakura::Private

#endif  // JETSTREAM_RENDER_SAKURA_STATE_HH
