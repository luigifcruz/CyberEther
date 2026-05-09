#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_ABOUT_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/about.hh"
#include "../../../views/modal/settings/components/about_info_table.hh"

#include "jetstream/backend/base.hh"
#include "jetstream/types.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct AboutPresenterInput {
    std::string viewportName;
    Extent2D<F32> displaySize;
    Extent2D<F32> framebufferScale;
    F32 renderScale = 1.0f;
};

inline std::vector<AboutInfoTable::Config> BuildAboutInfoTables(const AboutPresenterInput& input) {
    std::vector<AboutInfoTable::Config> tables;
    tables.push_back({
        .id = "AboutViewport",
        .title = "Viewport",
        .rows = {
            {"Adapter", input.viewportName},
            {"Window Size", jst::fmt::format("{:.0f} x {:.0f}", input.displaySize.x, input.displaySize.y)},
            {"Framebuffer Scale", jst::fmt::format("{:.2f} x {:.2f}",
                                                    input.framebufferScale.x,
                                                    input.framebufferScale.y)},
            {"Render Scale", jst::fmt::format("{:.2f}", input.renderScale)},
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

struct AboutPresenter {
    const PresenterContext& context;

    explicit AboutPresenter(const PresenterContext& context) : context(context) {}

    AboutSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return AboutSettingsPanel::Config{
            .updateAvailable = context.state.update.available,
            .checkingForUpdate = context.state.update.checking,
            .updateVersion = context.state.update.version,
            .accentKey = "cyber_blue",
            .infoTables = BuildAboutInfoTables({
                .viewportName = context.state.system.viewport->name(),
                .displaySize = context.state.system.viewport->displaySize(),
                .framebufferScale = context.state.system.render->framebufferScale(),
                .renderScale = context.state.system.render->scalingFactor(),
            }),
            .onCheckForUpdates = [enqueue]() {
                enqueue(MailCheckForUpdates{});
            },
            .onDownloadUpdate = []() {
                // TODO: Wire download logic.
            },
            .onDismissUpdate = [enqueue]() {
                enqueue(MailDismissUpdate{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_ABOUT_HH
