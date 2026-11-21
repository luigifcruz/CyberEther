#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_HH

#include "../context.hh"
#include "../agent/labels.hh"
#include "intelligence_model.hh"
#include "intelligence_provider.hh"
#include "../../model/messages.hh"
#include "../../views/settings/intelligence.hh"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceSettingsPresenter {
    const PresenterContext& context;
    IntelligenceProviderPresenter provider;
    IntelligenceModelPresenter model;

    explicit IntelligenceSettingsPresenter(const PresenterContext& context);
    static std::vector<std::string> ModelTags(const Settings::Intelligence::Model& model);
    static StatusBadge::Config Badge(const AgentProviderStatus& status);
    IntelligenceSettingsView::Config build() const;

    struct Options {
        std::vector<std::string> labels;
        std::optional<U64> selected;
    };

    template<typename Ids>
    static Options TitledOptions(const Ids& ids, const std::string& current) {
        Options options;
        for (const auto& id : ids) {
            if (id == current) {
                options.selected = options.labels.size();
            }
            options.labels.push_back(AgentLabels::Title(id));
        }
        return options;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_HH
