#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DOCS_SEARCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DOCS_SEARCH_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../support/docs.hh"
#include "../../views/modal/docs_search.hh"

namespace Jetstream {

struct DocsSearchModalPresenter {
    const PresenterContext& context;

    explicit DocsSearchModalPresenter(const PresenterContext& context);
    DocsSearchView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DOCS_SEARCH_HH
