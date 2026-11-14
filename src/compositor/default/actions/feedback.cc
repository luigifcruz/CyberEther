#include "feedback.hh"

namespace Jetstream {

FeedbackActions::FeedbackActions(DefaultCompositorState& state,
                                 DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

Result FeedbackActions::handle(const MailSetFeedbackText& msg) {
    state.feedback.text = msg.value;
    state.feedback.status = DefaultCompositorState::FeedbackState::Status::Idle;
    return Result::SUCCESS;
}

Result FeedbackActions::handle(const MailSubmitFeedback&) {
    const auto& text = state.feedback.text;

    if (text.size() < Feedback::MIN_LENGTH) {
        callbacks.notify(Sakura::ToastType::Error, 5000,
            jst::fmt::format("Feedback must be at least {} characters.", Feedback::MIN_LENGTH));
        return Result::SUCCESS;
    }

    if (text.size() > Feedback::MAX_LENGTH) {
        callbacks.notify(Sakura::ToastType::Error, 5000,
            jst::fmt::format("Feedback exceeds the {} character limit.", Feedback::MAX_LENGTH));
        return Result::SUCCESS;
    }

    state.feedback.status = DefaultCompositorState::FeedbackState::Status::Submitting;

    callbacks.submitFeedback(text);
    callbacks.enqueueMail(MailCloseModal{});

    return Result::SUCCESS;
}

}  // namespace Jetstream
