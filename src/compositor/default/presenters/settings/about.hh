#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_ABOUT_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/about.hh"
#include "../../views/settings/components/about_info_table.hh"

#include "jetstream/backend/base.hh"
#include "jetstream/config.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/types.hh"

#include <string>
#include <thread>
#include <vector>

namespace Jetstream {

struct AboutPresenterInput {
    std::string viewportName;
    DeviceType device = DeviceType::None;
    Render::WindowMetrics metrics;
    Extent2D<F32> contentScale = {1.0f, 1.0f};
    Render::WindowMetrics::Policy scalePolicy = Render::WindowMetrics::Policy::Interface;
    bool integratedTitlebar = false;
    bool retainedPresent = false;
    std::string dependencyPolicy;
};

inline std::string AboutThermalLabel(U64 state) {
    switch (state) {
        case 0: return "Nominal";
        case 1: return "Fair";
        case 2: return "Serious";
        case 3: return "Critical";
    }
    return "Unknown";
}

inline std::string AboutDeviceTypeLabel(Backend::PhysicalDeviceType type) {
    switch (type) {
        case Backend::PhysicalDeviceType::DISCRETE: return "Discrete";
        case Backend::PhysicalDeviceType::INTEGRATED: return "Integrated";
        case Backend::PhysicalDeviceType::OTHER: return "Other";
        case Backend::PhysicalDeviceType::UNKNOWN: break;
    }
    return "Unknown";
}

inline std::string AboutMemoryLabel(U64 bytes) {
    return jst::fmt::format("{:.0f} GB", static_cast<F64>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

inline std::string AboutPolicyLabel(const std::string& policy) {
    if (policy == "allow") {
        return "Install Automatically";
    }
    if (policy == "deny") {
        return "Never Install";
    }
    return "Ask Every Time";
}

inline std::vector<AboutInfoTable::Config> BuildAboutInfoTables(const AboutPresenterInput& input) {
    std::vector<AboutInfoTable::Config> tables;
    tables.push_back({
        .id = "AboutViewport",
        .title = "Viewport",
        .rows = {
            {"Adapter", input.viewportName},
            {"Renderer", GetDevicePrettyName(input.device)},
            {"Window Size", jst::fmt::format("{:.0f} x {:.0f}", input.metrics.displaySize.x, input.metrics.displaySize.y)},
            {"Framebuffer Size", jst::fmt::format("{:.0f} x {:.0f}",
                                                  input.metrics.framebufferSize.x,
                                                  input.metrics.framebufferSize.y)},
            {"Framebuffer Scale", jst::fmt::format("{:.2f} x {:.2f}",
                                                   input.metrics.framebufferScale.x,
                                                   input.metrics.framebufferScale.y)},
            {"Content Scale", jst::fmt::format("{:.2f} x {:.2f}", input.contentScale.x, input.contentScale.y)},
            {"Pixel Ratio", jst::fmt::format("{:.2f}", input.metrics.pixelRatio)},
            {"Interface Scale", jst::fmt::format("{:.2f}", input.metrics.scalingFactor)},
            {"Scale Policy", input.scalePolicy == Render::WindowMetrics::Policy::ContentScale ? "Content Scale"
                                                                                               : "Interface"},
            {"Titlebar", input.integratedTitlebar ? "Integrated" : "Native"},
            {"Presentation", input.retainedPresent ? "Retained" : "Continuous"},
        },
    });

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    if (Backend::Initialized<DeviceType::CPU>()) {
        auto& backend = Backend::State<DeviceType::CPU>();
        const auto hardwareThreads = std::thread::hardware_concurrency();
        const auto hardwareThreadsText = hardwareThreads == 0 ? std::string("Unknown") : std::to_string(hardwareThreads);
        const auto pythonRuntimePath = backend->getPythonRuntimePath().empty() ? std::string("Auto") : backend->getPythonRuntimePath();
        tables.push_back({
            .id = "AboutCpuBackend",
            .title = "CPU Backend",
            .rows = {
                {"Status", "Initialized"},
                {"Hardware Threads", hardwareThreadsText},
                {"Python Runtime", pythonRuntimePath},
                {"Dependency Policy", AboutPolicyLabel(input.dependencyPolicy)},
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
                {"Device Index", jst::fmt::format("{}", backend->getDeviceId())},
                {"API Version", backend->getApiVersion()},
                {"Compute Capability", backend->getComputeCapability()},
                {"Physical Memory", jst::fmt::format("{:.0f} GB", static_cast<float>(backend->getPhysicalMemory()) / (1024 * 1024 * 1024))},
                {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                {"RDMA VMM Support", backend->hasRdmaMemorySupport() ? "Yes" : "No"},
                {"Can Import Device Memory", backend->canImportDeviceMemory() ? "Yes" : "No"},
                {"Can Export Device Memory", backend->canExportDeviceMemory() ? "Yes" : "No"},
                {"Can Import Host Memory", backend->canImportHostMemory() ? "Yes" : "No"},
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
                {"Device Type", AboutDeviceTypeLabel(backend->getPhysicalDeviceType())},
                {"API Version", backend->getApiVersion()},
                {"Physical Memory", AboutMemoryLabel(backend->getPhysicalMemory())},
                {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                {"Processors", jst::fmt::format("{} of {} active", backend->getActiveProcessorCount(),
                                                backend->getTotalProcessorCount())},
                {"Multisampling", jst::fmt::format("{}x", backend->getMultisampling())},
                {"Low Power Mode", backend->getLowPowerStatus() ? "Yes" : "No"},
                {"Thermal State", AboutThermalLabel(backend->getThermalState())},
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
                {"Device Index", jst::fmt::format("{}", backend->getDeviceId())},
                {"API Version", backend->getApiVersion()},
                {"Physical Memory", jst::fmt::format("{:.0f} GB", static_cast<float>(backend->getPhysicalMemory()) / (1024 * 1024 * 1024))},
                {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                {"Can Import Device Memory", backend->canImportDeviceMemory() ? "Yes" : "No"},
                {"Can Export Device Memory", backend->canExportDeviceMemory() ? "Yes" : "No"},
                {"Can Import Host Memory", backend->canImportHostMemory() ? "Yes" : "No"},
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
        const auto& viewport = *context.state.system.viewport;

        return AboutSettingsPanel::Config{
            .updateSupported = context.state.update.supported,
            .updateUpToDate = context.state.update.upToDate,
            .updateFailed = context.state.update.failed,
            .updateChecking = context.state.update.checking,
            .updateAvailable = context.state.update.available,
            .updateDownloading = context.state.update.downloading,
            .updateReady = context.state.update.ready,
            .updateApplying = context.state.update.applying,
            .updateProgress = context.state.update.progress,
            .updateVersion = context.state.update.version,
            .updateReleaseNotes = context.state.update.releaseNotes,
            .updateMessage = context.state.update.message,
            .accentKey = "cyber_blue",
            .infoTables = BuildAboutInfoTables({
                .viewportName = viewport.name(),
                .device = viewport.device(),
                .metrics = context.state.system.render->metrics(),
                .contentScale = viewport.contentScale(),
                .scalePolicy = viewport.scalePolicy(),
                .integratedTitlebar = viewport.titlebarChrome().integrated,
                .retainedPresent = viewport.retainedPresent(),
                .dependencyPolicy = context.state.runtime.dependencyPolicy,
            }),
            .onCheckForUpdates = [enqueue]() {
                enqueue(MailCheckForUpdates{});
            },
            .onDownloadUpdate = [enqueue]() {
                enqueue(MailDownloadUpdate{});
            },
            .onApplyUpdate = [enqueue]() {
                enqueue(MailApplyUpdate{});
            },
            .onSendFeedback = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Feedback});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_ABOUT_HH
