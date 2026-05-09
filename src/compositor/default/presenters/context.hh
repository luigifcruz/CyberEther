#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_CONTEXT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_CONTEXT_HH

#include "../model/callbacks.hh"
#include "../model/state.hh"

namespace Jetstream {

struct PresenterContext {
    const DefaultCompositorState& state;
    const DefaultCompositorCallbacks& callbacks;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_CONTEXT_HH
