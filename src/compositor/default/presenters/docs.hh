#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_DOCS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_DOCS_HH

#include "context.hh"
#include "../support/docs.hh"
#include "../views/docs.hh"

namespace Jetstream {

struct DocsPresenter {
    const PresenterContext& context;

    explicit DocsPresenter(const PresenterContext& context) : context(context) {}

    DocsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& ui = context.state.interface;
        return {
            .docs = &Docs::Bundled(),
            .page = ui.docsPage,
            .anchor = ui.docsAnchor,
            .navigation = ui.docsNavigations,
            .onNavigate = [enqueue](const std::string& page, const std::string& anchor) {
                enqueue(MailOpenDocs{.page = page, .anchor = anchor});
            },
            .onSearch = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::DocsSearch});
            },
            .onOpenUrl = [enqueue](const std::string& url) {
                enqueue(MailOpenUrl{.url = url, .notifyResult = true});
            },
            .onFeedback = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Feedback});
            },
            .onClose = [enqueue]() {
                enqueue(MailCloseDocs{});
            },
            .onFocus = [enqueue]() {
                enqueue(MailFocusDocs{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_DOCS_HH
