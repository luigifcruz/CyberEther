#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH

#include "jetstream/detail/compositor_impl.hh"

#include "updater.hh"
#include "feedback.hh"

#include "actions/base.hh"
#include "model/callbacks.hh"
#include "model/state.hh"
#include "presenters/base.hh"
#include "views/workbench.hh"

#include <deque>
#include <string>

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
    Updater updater;
    Feedback feedback;
    DefaultCompositorCallbacks callbacks;
    DefaultActions actions;
    DefaultPresenterRegistry presenters;

    std::deque<Mail> pendingMail;
    WorkbenchView workbench;

    void enqueue(Mail&& mail);

    // Workbench.

    void updateWorkbenchState();
    void updateFilePendingState();
    void updateDependencyState();
    void updateBenchmarkState();
    void updateRemoteState();
    void updateUpdaterState();
    void updateFeedbackState();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH
