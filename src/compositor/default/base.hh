#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH

#include "jetstream/detail/compositor_impl.hh"

#include "actions/base.hh"
#include "model/callbacks.hh"
#include "model/state.hh"
#include "presenters/base.hh"
#include "views/workbench.hh"

#include <deque>
#include <string>
#include <vector>

namespace Jetstream {

class DefaultCompositor : public Compositor::Impl {
 public:
    DefaultCompositor();

    Result create();
    Result destroy();

    Result present();
    Result poll();

 private:
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    DefaultActions actions;
    DefaultPresenterRegistry presenters;

    std::deque<Mail> pendingMail;
    std::vector<std::string> flowgraphIds;
    WorkbenchView workbench;

    void enqueue(Mail&& mail);

    // Workbench.

    void updateWorkbenchState();
    void updateFilePendingState();
    void updateBenchmarkState();
    void updateRemoteState();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH
