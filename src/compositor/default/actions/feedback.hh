#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../support/feedback.hh"

#include "jetstream/logger.hh"

#include <tuple>

namespace Jetstream {

struct FeedbackActions {
    using Filter = std::tuple<MailSetFeedbackText,
                              MailSubmitFeedback>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    FeedbackActions(DefaultCompositorState& state,
                    DefaultCompositorCallbacks& callbacks);
    Result handle(const MailSetFeedbackText& msg);
    Result handle(const MailSubmitFeedback&);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH
