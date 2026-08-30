#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/logger.hh"

#include <tuple>

namespace Jetstream {

struct FeedbackActions {
    using Filter = std::tuple<MailSetFeedbackText,
                              MailSubmitFeedback>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    FeedbackActions(DefaultCompositorState& state,
                    DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result handle(const MailSetFeedbackText& msg) {
        state.feedback.text = msg.value;
        state.feedback.status = DefaultCompositorState::FeedbackState::Status::Idle;
        return Result::SUCCESS;
    }

    Result handle(const MailSubmitFeedback&) {
        const auto& text = state.feedback.text;

        if (text.size() < Feedback::MIN_LENGTH) {
            callbacks.notify(Sakura::ToastType::Error, 5000,
                "Feedback must be at least " +
                std::to_string(Feedback::MIN_LENGTH) + " characters.");
            return Result::SUCCESS;
        }

        if (text.size() > Feedback::MAX_LENGTH) {
            callbacks.notify(Sakura::ToastType::Error, 5000,
                "Feedback exceeds the " +
                std::to_string(Feedback::MAX_LENGTH) + " character limit.");
            return Result::SUCCESS;
        }

        state.feedback.status = DefaultCompositorState::FeedbackState::Status::Submitting;

        callbacks.submitFeedback(text);
        callbacks.enqueueMail(MailCloseModal{});

        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_FEEDBACK_HH
