#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../support/feedback.hh"

#include "jetstream/logger.hh"


namespace Jetstream {

struct FeedbackActions {
    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    static std::string FormatBlockDebugInfo(const Flowgraph::View::BlockData& block);

    FeedbackActions(DefaultCompositorState& state,
                    DefaultCompositorCallbacks& callbacks);
    Result handle(const MailSetFeedbackText& msg);
    Result handle(const MailSetFeedbackDebugInfo& msg);
    Result handle(const MailReportBlockProblem& msg);
    Result handle(const MailSubmitFeedback&);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH
