#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DOCS_SEARCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DOCS_SEARCH_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../support/docs.hh"
#include "../../views/modal/docs_search.hh"

namespace Jetstream {

struct DocsSearchModalPresenter {
    const PresenterContext& context;

    explicit DocsSearchModalPresenter(const PresenterContext& context) : context(context) {}

    DocsSearchView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return {
            .docs = &Docs::Bundled(),
            .query = context.state.interface.docsQuery,
            .onQueryChange = [enqueue](const std::string& query) {
                enqueue(MailSetDocsQuery{.query = query});
            },
            .onOpen = [enqueue](const std::string& page, const std::string& anchor) {
                enqueue(MailCloseModal{});
                enqueue(MailOpenDocs{.page = page, .anchor = anchor});
            },
            .onClose = [enqueue]() {
                enqueue(MailCloseModal{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DOCS_SEARCH_HH
