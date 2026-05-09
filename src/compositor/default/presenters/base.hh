#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BASE_HH

#include "context.hh"
#include "workbench.hh"

namespace Jetstream {

struct DefaultPresenterRegistry {
 public:
    DefaultPresenterRegistry(const DefaultCompositorState& state,
                             const DefaultCompositorCallbacks& callbacks) : context{state, callbacks},
                                                                             workbench(context) {}

    WorkbenchView::Config build() const {
        return workbench.build();
    }

 private:
    PresenterContext context;
    WorkbenchPresenter workbench;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BASE_HH
