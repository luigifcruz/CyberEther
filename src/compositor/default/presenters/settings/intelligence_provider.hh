#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_PROVIDER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_PROVIDER_HH

#include "../context.hh"
#include "../../support/agent/providers.hh"
#include "../../model/messages.hh"
#include "../../views/settings/intelligence_provider.hh"

#include <optional>
#include <string>

namespace Jetstream {

struct IntelligenceProviderPresenter {
    static constexpr Agent::Provider::Kind DefaultKind = Agent::Provider::Kind::OpenRouter;

    const PresenterContext& context;

    explicit IntelligenceProviderPresenter(const PresenterContext& context);
    IntelligenceProviderView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_PROVIDER_HH
