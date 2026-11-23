#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLAN_NOTICE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLAN_NOTICE_HH

#include "../components/card_button.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <utility>

namespace Jetstream {

struct PlanNoticeView {
    static constexpr F32 Width = 460.0f;
    static constexpr F32 ButtonHeight = 40.0f;
    static constexpr F32 IconScale = 2.4f;

    struct Config {
        Sakura::Callback<void()> onManageUsage;
        Sakura::Callback<void()> onAddModels;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    Sakura::Spacer topSpacing;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Stack header;
    Sakura::Spacer middleSpacing;
    CardButton manage;
    CardButton addModels;
    Sakura::Grid footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLAN_NOTICE_HH
