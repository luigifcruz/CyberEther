#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SECTION_GROUP_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SECTION_GROUP_HH

#include "callout.hh"

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct SectionGroup : public Sakura::Stack {
    static constexpr F32 CalloutGap = Sakura::Spacing::XSmall;

    struct Config {
        std::string title;
        std::string description;
        std::string warning;
        std::vector<Sakura::SettingCard::Row> rows;
        std::string empty;
        Sakura::Component* action = nullptr;
        Sakura::Component* toolbar = nullptr;
        Sakura::Component* footer = nullptr;
        bool trailing = true;
    };

    void update(Config config);

 private:
    Sakura::Text titleText;
    Sakura::Stack textColumn;
    Sakura::Spacer titleSpacing;
    Sakura::Stack titleRow;
    Sakura::Text descriptionText;
    Sakura::Spacer warningSpacing;
    Callout warningCallout;
    Sakura::Spacer toolbarSpacing;
    Sakura::Spacer cardSpacing;
    Sakura::SettingCard card;
    Sakura::Spacer footerSpacing;
    Sakura::Spacer trailing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SECTION_GROUP_HH
