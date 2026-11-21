#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_LEGAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_LEGAL_HH

#include "../components/compact_button.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct LegalSettingsView {
    struct Config {
        Sakura::Callback<void()> onViewFullLicenses;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    struct Dependency {
        std::string name;
        std::string license;
    };

    static const std::vector<Dependency>& Dependencies();

    Config config;
    SettingsHeader header;
    SectionGroup licenseGroup;
    std::vector<std::unique_ptr<Sakura::Badge>> licenses;
    CompactButton fullLicensesButton;
    SectionGroup thirdPartyGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_LEGAL_HH
