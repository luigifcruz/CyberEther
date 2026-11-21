#include "general.hh"

namespace Jetstream {

GeneralSettingsPresenter::GeneralSettingsPresenter(const PresenterContext& context) : context(context) {}

GeneralSettingsView::Config GeneralSettingsPresenter::build() const {
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
        .touchpadScrollSpeed = context.state.interface.touchpadScrollSpeed,
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
        .onTouchpadScrollSpeedChange = [enqueue](F32 value) {
            enqueue(MailSetTouchpadScrollSpeed{.value = value});
        },
    };
}

}  // namespace Jetstream
