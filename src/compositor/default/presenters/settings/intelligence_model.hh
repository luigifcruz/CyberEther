#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_MODEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_MODEL_HH

#include "../context.hh"
#include "../../model/messages.hh"
#include "../agent/labels.hh"
#include "../../views/settings/intelligence_model.hh"

#include "tools/text.hh"
#include "tools/value.hh"

#include <string>

namespace Jetstream {

struct IntelligenceModelPresenter {
    static constexpr U64 MaxShown = 100;

    const PresenterContext& context;

    explicit IntelligenceModelPresenter(const PresenterContext& context);
    IntelligenceModelView::Config build() const;

 private:
    std::string status(U64 shown, U64 matched) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_MODEL_HH
