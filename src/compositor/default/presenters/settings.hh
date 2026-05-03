#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../themes.hh"
#include "../views/modal/settings/base.hh"
#include "../views/modal/settings/components/about_info_table.hh"

#include "jetstream/backend/base.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct DefaultSettingsPresenter {
    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    std::vector<AboutInfoTable::Config> buildAboutTables() const {
        std::vector<AboutInfoTable::Config> tables;
        tables.push_back({
            .id = "AboutViewport",
            .title = "Viewport",
            .rows = {
                {"Adapter", state.system.viewport->name()},
                {"Window Size", jst::fmt::format("{:.0f} x {:.0f}",
                                                  state.system.viewport->displaySize().x,
                                                  state.system.viewport->displaySize().y)},
                {"Framebuffer Scale", jst::fmt::format("{:.2f} x {:.2f}",
                                                        state.system.render->framebufferScale().x,
                                                        state.system.render->framebufferScale().y)},
                {"Render Scale", jst::fmt::format("{:.2f}", state.system.render->scalingFactor())},
            },
        });

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        if (Backend::Initialized<DeviceType::CPU>()) {
            tables.push_back({
                .id = "AboutCpuBackend",
                .title = "CPU Backend",
                .rows = {
                    {"Status", "Initialized"},
                },
            });
        }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (Backend::Initialized<DeviceType::CUDA>()) {
            auto& backend = Backend::State<DeviceType::CUDA>();
            tables.push_back({
                .id = "AboutCudaBackend",
                .title = "CUDA Backend",
                .rows = {
                    {"Device", backend->getDeviceName()},
                    {"API Version", backend->getApiVersion()},
                    {"Compute Capability", backend->getComputeCapability()},
                    {"Physical Memory", jst::fmt::format("{:.0f} GB", static_cast<float>(backend->getPhysicalMemory()) / (1024 * 1024 * 1024))},
                    {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                    {"Export Device Memory", backend->canExportDeviceMemory() ? "Yes" : "No"},
                    {"Import Device Memory", backend->canImportDeviceMemory() ? "Yes" : "No"},
                    {"Import Host Memory", backend->canImportHostMemory() ? "Yes" : "No"},
                },
            });
        }
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        if (Backend::Initialized<DeviceType::Metal>()) {
            auto& backend = Backend::State<DeviceType::Metal>();
            tables.push_back({
                .id = "AboutMetalBackend",
                .title = "Metal Backend",
                .rows = {
                    {"Device", backend->getDeviceName()},
                    {"API Version", backend->getApiVersion()},
                    {"Physical Memory", jst::fmt::format("{:.0f} GB", static_cast<float>(backend->getPhysicalMemory()) / (1024 * 1024 * 1024))},
                    {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                    {"Processors", jst::fmt::format("{}", backend->getActiveProcessorCount())},
                    {"Low Power Mode", backend->getLowPowerStatus() ? "Yes" : "No"},
                    {"Thermal State", jst::fmt::format("{}", backend->getThermalState())},
                },
            });
        }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        if (Backend::Initialized<DeviceType::Vulkan>()) {
            auto& backend = Backend::State<DeviceType::Vulkan>();
            tables.push_back({
                .id = "AboutVulkanBackend",
                .title = "Vulkan Backend",
                .rows = {
                    {"Device", backend->getDeviceName()},
                    {"API Version", backend->getApiVersion()},
                    {"Physical Memory", jst::fmt::format("{:.0f} GB", static_cast<float>(backend->getPhysicalMemory()) / (1024 * 1024 * 1024))},
                    {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                    {"Low Power Mode", backend->getLowPowerStatus() ? "Yes" : "No"},
                    {"Thermal State", jst::fmt::format("{}", backend->getThermalState())},
                },
            });
        }
#endif

#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
        if (Backend::Initialized<DeviceType::WebGPU>()) {
            auto& backend = Backend::State<DeviceType::WebGPU>();
            tables.push_back({
                .id = "AboutWebGpuBackend",
                .title = "WebGPU Backend",
                .rows = {
                    {"Device", backend->getDeviceName()},
                    {"API Version", backend->getApiVersion()},
                    {"Physical Memory", jst::fmt::format("{:.0f} GB", static_cast<float>(backend->getPhysicalMemory()) / (1024 * 1024 * 1024))},
                    {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                    {"Low Power Mode", backend->getLowPowerStatus() ? "Yes" : "No"},
                    {"Thermal State", jst::fmt::format("{}", backend->getThermalState())},
                },
            });
        }
#endif

        return tables;
    }

    SettingsView::Config build() const {
        const auto enqueue = callbacks.enqueueMail;
        return SettingsView::Config{
            .section = state.settings.section,
            .onSectionChange = [enqueue](DefaultCompositorState::SettingsState::Section section) {
                enqueue(MailSetSettingsSection{.section = section});
            },
            .general = {
                .themes = BuildThemeKeys(),
                .currentThemeKey = state.sakura.themeKey,
                .interfaceScale = state.system.render->scalingFactor(),
                .renderer = GetDevicePrettyName(state.system.render->device()),
                .infoPanelEnabled = state.interface.infoPanelEnabled,
                .backgroundParticles = state.interface.backgroundParticles,
                .onThemeChange = [enqueue](const std::string& themeKey) {
                    enqueue(MailApplyTheme{themeKey});
                },
                .onInfoPanelChange = [enqueue](bool value) {
                    enqueue(MailSetInfoPanelEnabled{.value = value});
                },
                .onBackgroundParticlesChange = [enqueue](bool value) {
                    enqueue(MailSetBackgroundParticles{.value = value});
                },
            },
            .remote = {
                .started = state.remote.started,
                .brokerUrl = state.remote.brokerUrl,
                .codec = state.remote.codec,
                .framerate = state.remote.framerate,
                .encoder = state.remote.encoder,
                .autoJoinSessions = state.remote.autoJoinSessions,
                .onBrokerUrlChange = [enqueue](const std::string& value) {
                    enqueue(MailSetRemoteBrokerUrl{.value = value});
                },
                .onCodecChange = [enqueue](Instance::Remote::CodecType value) {
                    enqueue(MailSetRemoteCodec{.value = value});
                },
                .onFramerateChange = [enqueue](U32 value) {
                    enqueue(MailSetRemoteFramerate{.value = value});
                },
                .onEncoderChange = [enqueue](Instance::Remote::EncoderType value) {
                    enqueue(MailSetRemoteEncoder{.value = value});
                },
                .onAutoJoinSessionsChange = [enqueue](bool value) {
                    enqueue(MailSetRemoteAutoJoinSessions{.value = value});
                },
            },
            .developer = {
                .latencyEnabled = state.debug.latencyEnabled,
                .runtimeMetricsEnabled = state.debug.runtimeMetricsEnabled,
                .logLevel = state.debug.logLevel,
                .onLatencyEnabledChange = [enqueue](bool value) {
                    enqueue(MailSetDebugLatencyEnabled{.value = value});
                },
                .onRuntimeMetricsEnabledChange = [enqueue](bool value) {
                    enqueue(MailSetDebugRuntimeMetricsEnabled{.value = value});
                },
                .onLogLevelChange = [enqueue](int value) {
                    enqueue(MailSetDebugLogLevel{.value = value});
                },
            },
            .about = {
                .updateAvailable = state.update.available,
                .checkingForUpdate = state.update.checking,
                .updateVersion = state.update.version,
                .accentKey = "cyber_blue",
                .infoTables = buildAboutTables(),
                .onCheckForUpdates = [enqueue]() {
                    enqueue(MailCheckForUpdates{});
                },
                .onDownloadUpdate = []() {
                    // TODO: Wire download logic.
                },
                .onDismissUpdate = [enqueue]() {
                    enqueue(MailDismissUpdate{});
                },
            },
            .legal = {
                .onViewFullLicenses = [enqueue]() {
                    enqueue(MailOpenUrl{.url = "https://cyberether.org/docs/acknowledgments"});
                },
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_HH
