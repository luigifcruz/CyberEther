#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_FEEDBACK_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/feedback.hh"

#include <string>

namespace Jetstream {

struct FeedbackModalPresenter {
    const PresenterContext& context;

    explicit FeedbackModalPresenter(const PresenterContext& context) : context(context) {}

    FeedbackView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const std::string text = context.state.feedback.text;
        const auto status = context.state.feedback.status;

        FeedbackView::Config::Status viewStatus = FeedbackView::Config::Status::Idle;
        switch (status) {
            case DefaultCompositorState::FeedbackState::Status::Idle:
                viewStatus = FeedbackView::Config::Status::Idle;
                break;
            case DefaultCompositorState::FeedbackState::Status::Submitting:
                viewStatus = FeedbackView::Config::Status::Submitting;
                break;
        }

        return FeedbackView::Config{
            .text = text,
            .minLength = Feedback::MIN_LENGTH,
            .maxLength = Feedback::MAX_LENGTH,
            .status = viewStatus,
            .onTextChange = [enqueue](const std::string& value) {
                enqueue(MailSetFeedbackText{.value = value});
            },
            .onSubmit = [enqueue]() {
                enqueue(MailSubmitFeedback{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_FEEDBACK_HH
