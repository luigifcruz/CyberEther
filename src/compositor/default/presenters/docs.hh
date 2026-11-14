#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_DOCS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_DOCS_HH

#include "context.hh"
#include "../support/docs.hh"
#include "../views/docs.hh"

namespace Jetstream {

struct DocsPresenter {
    const PresenterContext& context;

    explicit DocsPresenter(const PresenterContext& context);
    DocsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_DOCS_HH
