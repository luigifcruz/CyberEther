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

    explicit IntelligenceModelPresenter(const PresenterContext& context) : context(context) {}

    IntelligenceModelPage::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& catalog = context.state.agent.catalog;

        IntelligenceModelPage::Config config;
        config.search = catalog.search;

        const std::string needle = Text::ToLower(catalog.search);
        U64 matched = 0;
        for (const auto& model : catalog.models) {
            const bool matches = Value::KeyMatches(model.id, needle) ||
                                 Value::KeyMatches(model.name, needle) ||
                                 Value::KeyMatches(model.provider, needle);
            if (!matches) {
                continue;
            }
            if (++matched <= MaxShown) {
                config.models.push_back({.model = model, .label = AgentLabels::Model(model)});
            }
        }

        config.status = status(config.models.size(), matched);
        config.warning = !catalog.loading && !catalog.error.empty();

        config.onSearch = [enqueue](const std::string& value) {
            enqueue(MailSetAgentModelSearch{.value = value});
        };
        config.onCommit = [enqueue](const Settings::Intelligence::Model& model) {
            enqueue(MailAddAgentModel{.value = model});
        };
        config.onBack = [enqueue]() {
            enqueue(MailSetIntelligencePage{.page = IntelligencePage::Overview});
        };

        return config;
    }

 private:
    std::string status(U64 shown, U64 matched) const {
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
        if (shown < matched) {
            return jst::fmt::format("Showing {} of {} matches. Refine the filter to see the rest.", shown, matched);
        }
        return jst::fmt::format("Showing {} of {} models.", shown, catalog.models.size());
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_MODEL_HH
