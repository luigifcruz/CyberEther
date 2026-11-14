#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_MODEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_MODEL_HH

#include "../components/compact_button.hh"
#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/settings.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceModelView {
    using Option = Settings::Intelligence::Model;

    struct Item {
        Option model;
        std::string label;
    };

    struct Config {
        std::string search;
        std::vector<Item> models;
        std::string status;
        bool warning = false;
        Sakura::Callback<void(const std::string&)> onSearch;
        Sakura::Callback<void(const Option&)> onCommit;
        Sakura::Callback<void()> onBack;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    struct Entry {
        StatusBadge id;
        CompactButton add;
        Sakura::Stack row;
    };

    Config config;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    Sakura::PillInput filterInput;
    std::vector<std::unique_ptr<Entry>> entries;
    SectionGroup modelsGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_MODEL_HH
