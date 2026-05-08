#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_INFO_HH

#include "../context.hh"

#include "../../views/hud/info.hh"

namespace Jetstream {

struct InfoHudPresenter {
    const PresenterContext& context;

    explicit InfoHudPresenter(const PresenterContext& context) : context(context) {}

    InfoHudView::Config build() const {
        return InfoHudView::Config{
            .visible = context.state.interface.infoPanelEnabled,
            .frameRate = Sakura::FrameRate(),
            .viewportName = context.state.system.viewport->name(),
            .renderInfo = context.state.system.render->info(),
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_INFO_HH
