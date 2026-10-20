#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH

#include "jetstream/detail/compositor_impl.hh"

#include "updater.hh"
#include "support/feedback.hh"

#include "actions/base.hh"
#include "support/agent/runtime.hh"
#include "model/callbacks.hh"
#include "model/state.hh"
#include "presenters/base.hh"
#include "views/workbench.hh"

#include <deque>
#include <optional>
#include <string>

namespace Jetstream {

class DefaultCompositor : public Compositor::Impl {
 public:
    DefaultCompositor();
    ~DefaultCompositor();

    Result create();
    Result destroy();

    Result present();
    Result poll();

 private:
    DefaultCompositorState state;
    Updater updater;
    Feedback feedback;
    DefaultCompositorCallbacks callbacks;
    AgentRuntime agent;
    DefaultActions actions;
    DefaultPresenterRegistry presenters;

    std::deque<Mail> pendingMail;
    WorkbenchView workbench;
    std::optional<bool> previousIntegratedTitlebar;

    void enqueue(Mail&& mail);

    // Workbench.

    void updateWorkbenchState();
    void updateFlowgraphBlockState();
    void updateFilePendingState();
    void updateDependencyState();
    void updateBenchmarkState();
    void updateRemoteState();
    void updateUpdaterState();
    void updateFeedbackState();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_HH
