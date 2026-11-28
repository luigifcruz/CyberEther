#include "general.hh"

#include "jetstream/backend/base.hh"

#include <vector>

namespace Jetstream {

namespace {

std::vector<Backend::PhysicalDevice> GpuDevices(DeviceType renderer) {
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (renderer == DeviceType::Vulkan && Backend::Initialized<DeviceType::Vulkan>()) {
        return Backend::State<DeviceType::Vulkan>()->getDevices();
    }
#else
    (void)renderer;
#endif
    return {};
}

}  // namespace

GeneralSettingsPresenter::GeneralSettingsPresenter(const PresenterContext& context) : context(context) {}

GeneralSettingsView::Config GeneralSettingsPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto renderer = context.state.system.render ? context.state.system.render->device() : DeviceType::None;
    const auto gpu = GpuLabels::Build(GpuDevices(renderer), context.state.graphics.gpu);
    std::vector<std::string> gpuOptions;
    std::vector<std::string> gpuDetails;
    std::vector<Settings::Gpu> gpuValues;
    for (const auto& option : gpu.options) {
        gpuOptions.push_back(option.label);
        gpuDetails.push_back(option.detail);
        gpuValues.push_back(option.value);
    }
    return GeneralSettingsView::Config{
        .themes = Sakura::BuildThemeKeys(),
        .currentThemeKey = context.state.sakura.themeKey,
        .interfaceScale = context.state.graphics.scale,
        .renderer = context.state.graphics.device.has_value()
            ? GetDevicePrettyName(context.state.graphics.device.value())
            : GetDevicePrettyName(context.state.system.render->device()),
        .gpuVisible = renderer == DeviceType::Vulkan,
        .gpuOptions = std::move(gpuOptions),
        .gpuDetails = std::move(gpuDetails),
        .gpuSelected = gpu.selected,
        .gpuDisabled = gpu.disabled,
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
        .onGpuSelect = [enqueue, values = std::move(gpuValues)](U64 index) {
            if (index < values.size()) {
                enqueue(MailSetGraphicsGpu{.value = values[index]});
            }
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
