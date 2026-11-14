#include "base.hh"

namespace Jetstream {

DefaultPresenterRegistry::DefaultPresenterRegistry(const DefaultCompositorState& state,
                                                   const DefaultCompositorCallbacks& callbacks) : context{state, callbacks},
                                                                                                  workbench(context) {}

WorkbenchView::Config DefaultPresenterRegistry::build() const {
    return workbench.build();
}

}  // namespace Jetstream
