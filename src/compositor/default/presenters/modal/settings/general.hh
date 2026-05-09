#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_GENERAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_GENERAL_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../themes.hh"
#include "../../../views/modal/settings/general.hh"

#include <string>

namespace Jetstream {

struct GeneralSettingsPresenter {
    const PresenterContext& context;

    explicit GeneralSettingsPresenter(const PresenterContext& context) : context(context) {}

    GeneralSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return GeneralSettingsPanel::Config{
            .themes = BuildThemeKeys(),
            .currentThemeKey = context.state.sakura.themeKey,
            .interfaceScale = context.state.graphics.scale,
            .renderer = context.state.graphics.device.has_value()
                ? GetDevicePrettyName(context.state.graphics.device.value())
                : GetDevicePrettyName(context.state.system.render->device()),
            .framerate = context.state.graphics.framerate,
            .infoPanelEnabled = context.state.interface.infoPanelEnabled,
            .backgroundParticles = context.state.interface.backgroundParticles,
            .onThemeChange = [enqueue](const std::string& themeKey) {
                enqueue(MailApplyTheme{themeKey});
            },
            .onInterfaceScaleChange = [enqueue](F32 value) {
                enqueue(MailSetGraphicsScale{.value = value});
            },
            .onRendererChange = [enqueue](DeviceType value) {
                enqueue(MailSetGraphicsDevice{.value = value});
            },
            .onFramerateChange = [enqueue](U64 value) {
                enqueue(MailSetGraphicsFramerate{.value = value});
            },
            .onInfoPanelChange = [enqueue](bool value) {
                enqueue(MailSetInfoPanelEnabled{.value = value});
            },
            .onBackgroundParticlesChange = [enqueue](bool value) {
                enqueue(MailSetBackgroundParticles{.value = value});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_GENERAL_HH
