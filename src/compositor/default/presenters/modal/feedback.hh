#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_FEEDBACK_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/feedback.hh"
#include "../../support/feedback.hh"

#include <string>

namespace Jetstream {

struct FeedbackModalPresenter {
    const PresenterContext& context;

    explicit FeedbackModalPresenter(const PresenterContext& context);
    FeedbackView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_FEEDBACK_HH
