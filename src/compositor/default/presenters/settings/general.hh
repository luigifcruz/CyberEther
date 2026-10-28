#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GENERAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GENERAL_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/general.hh"

#include "render/sakura/themes.hh"

#include <string>

namespace Jetstream {

struct GeneralSettingsPresenter {
    const PresenterContext& context;

    explicit GeneralSettingsPresenter(const PresenterContext& context) : context(context) {}

    GeneralSettingsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return GeneralSettingsView::Config{
            .themes = Sakura::BuildThemeKeys(),
            .currentThemeKey = context.state.sakura.themeKey,
            .interfaceScale = context.state.graphics.scale,
            .renderer = context.state.graphics.device.has_value()
                ? GetDevicePrettyName(context.state.graphics.device.value())
                : GetDevicePrettyName(context.state.system.render->device()),
            .backgroundParticles = context.state.interface.backgroundParticles,
            .showTips = context.state.interface.showTips,
            .onThemeChange = [enqueue](const std::string& themeKey) {
                enqueue(MailApplyTheme{themeKey});
            },
            .onInterfaceScaleChange = [enqueue](F32 value) {
                enqueue(MailSetGraphicsScale{.value = value});
            },
            .onRendererChange = [enqueue](DeviceType value) {
                enqueue(MailSetGraphicsDevice{.value = value});
            },
            .onBackgroundParticlesChange = [enqueue](bool value) {
                enqueue(MailSetBackgroundParticles{.value = value});
            },
            .onShowTipsChange = [enqueue](bool value) {
                enqueue(MailSetShowTips{.value = value});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GENERAL_HH
