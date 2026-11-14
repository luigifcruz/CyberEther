#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_BASE_HH

#include "welcome.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/start_tab/base.hh"

#include <optional>

namespace Jetstream {

struct StartTabPresenter {
    const PresenterContext& context;
    WelcomePresenter welcome;

    explicit StartTabPresenter(const PresenterContext& context);
    std::optional<StartTabView::Config> build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_BASE_HH
