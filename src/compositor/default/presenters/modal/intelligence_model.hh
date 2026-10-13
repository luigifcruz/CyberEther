#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_INTELLIGENCE_MODEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_INTELLIGENCE_MODEL_HH

#include "../context.hh"
#include "../../model/messages.hh"
#include "../../views/modal/intelligence_model.hh"

#include "tools/text.hh"
#include "tools/value.hh"

#include <string>

namespace Jetstream {

struct IntelligenceModelPresenter {
    const PresenterContext& context;

    explicit IntelligenceModelPresenter(const PresenterContext& context) : context(context) {}

    IntelligenceModelView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& catalog = context.state.agent.catalog;

        IntelligenceModelView::Config config;
        config.search = catalog.search;

        const std::string needle = Text::ToLower(catalog.search);
        for (const auto& model : catalog.models) {
            const bool matches = Value::KeyMatches(model.id, needle) ||
                                 Value::KeyMatches(model.name, needle) ||
                                 Value::KeyMatches(model.provider, needle);
            if (matches) {
                config.models.push_back(model);
            }
        }

        config.status = status(config.models.size());
        config.warning = !catalog.loading && !catalog.error.empty();

        config.onSearch = [enqueue](const std::string& value) {
            enqueue(MailSetAgentModelSearch{.value = value});
        };
        config.onCommit = [enqueue](const Settings::Intelligence::Model& model) {
            enqueue(MailAddAgentModel{.value = model});
        };

        return config;
    }

 private:
    std::string status(U64 shown) const {
        const auto& catalog = context.state.agent.catalog;
        if (catalog.loading) {
            return "Fetching model lists...";
        }
        if (!catalog.error.empty()) {
            return catalog.error;
        }
        if (catalog.models.empty()) {
            return "No models were returned by the configured providers.";
        }
        return jst::fmt::format("{} of {} models shown.", shown, catalog.models.size());
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_INTELLIGENCE_MODEL_HH
