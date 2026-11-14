#include "docs_search.hh"

namespace Jetstream {

DocsSearchModalPresenter::DocsSearchModalPresenter(const PresenterContext& context) : context(context) {}

DocsSearchView::Config DocsSearchModalPresenter::build() const {
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

}  // namespace Jetstream
