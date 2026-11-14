#include "about.hh"

namespace Jetstream {

AboutPresenter::AboutPresenter(const PresenterContext& context) : context(context) {}

AboutSettingsView::Config AboutPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& viewport = *context.state.system.viewport;

    return AboutSettingsView::Config{
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
        .infoTables = InfoTables({
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

std::string AboutPresenter::ThermalLabel(U64 state) {
    switch (state) {
        case 0: return "Nominal";
        case 1: return "Fair";
        case 2: return "Serious";
        case 3: return "Critical";
    }
    return "Unknown";
}

std::string AboutPresenter::DeviceTypeLabel(Backend::PhysicalDeviceType type) {
    switch (type) {
        case Backend::PhysicalDeviceType::DISCRETE: return "Discrete";
        case Backend::PhysicalDeviceType::INTEGRATED: return "Integrated";
        case Backend::PhysicalDeviceType::OTHER: return "Other";
        case Backend::PhysicalDeviceType::UNKNOWN: break;
    }
    return "Unknown";
}

std::string AboutPresenter::MemoryLabel(U64 bytes) {
    return jst::fmt::format("{:.0f} GB", static_cast<F64>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

std::string AboutPresenter::PolicyLabel(const std::string& policy) {
    if (policy == "allow") {
        return "Install Automatically";
    }
    if (policy == "deny") {
        return "Never Install";
    }
    return "Ask Every Time";
}

std::vector<AboutInfoTable::Config> AboutPresenter::InfoTables(const Input& input) {
    std::vector<AboutInfoTable::Config> tables;
    tables.push_back({
        .title = "Viewport",
        .rows = {
            {"Adapter", input.viewportName},
            {"Renderer", GetDevicePrettyName(input.device)},
            {"Window Size", jst::fmt::format("{:.0f} x {:.0f}",
                                             input.metrics.displaySize.x,
                                             input.metrics.displaySize.y)},
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
        const auto hardwareThreadsText = hardwareThreads == 0 ? std::string("Unknown")
                                                              : jst::fmt::format("{}", hardwareThreads);
        const auto pythonRuntimePath = backend->getPythonRuntimePath().empty() ? std::string("Auto")
                                                                               : backend->getPythonRuntimePath();
        tables.push_back({
            .title = "CPU Backend",
            .rows = {
                {"Status", "Initialized"},
                {"Hardware Threads", hardwareThreadsText},
                {"Python Runtime", pythonRuntimePath},
                {"Dependency Policy", PolicyLabel(input.dependencyPolicy)},
            },
        });
    }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if (Backend::Initialized<DeviceType::CUDA>()) {
        auto& backend = Backend::State<DeviceType::CUDA>();
        tables.push_back({
            .title = "CUDA Backend",
            .rows = {
                {"Device", backend->getDeviceName()},
                {"Device Index", jst::fmt::format("{}", backend->getDeviceId())},
                {"API Version", backend->getApiVersion()},
                {"Compute Capability", backend->getComputeCapability()},
                {"Physical Memory", MemoryLabel(backend->getPhysicalMemory())},
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
            .title = "Metal Backend",
            .rows = {
                {"Device", backend->getDeviceName()},
                {"Device Type", DeviceTypeLabel(backend->getPhysicalDeviceType())},
                {"API Version", backend->getApiVersion()},
                {"Physical Memory", MemoryLabel(backend->getPhysicalMemory())},
                {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
                {"Processors", jst::fmt::format("{} of {} active", backend->getActiveProcessorCount(),
                                                backend->getTotalProcessorCount())},
                {"Multisampling", jst::fmt::format("{}x", backend->getMultisampling())},
                {"Low Power Mode", backend->getLowPowerStatus() ? "Yes" : "No"},
                {"Thermal State", ThermalLabel(backend->getThermalState())},
            },
        });
    }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (Backend::Initialized<DeviceType::Vulkan>()) {
        auto& backend = Backend::State<DeviceType::Vulkan>();
        tables.push_back({
            .title = "Vulkan Backend",
            .rows = {
                {"Device", backend->getDeviceName()},
                {"Device Index", jst::fmt::format("{}", backend->getDeviceId())},
                {"API Version", backend->getApiVersion()},
                {"Physical Memory", MemoryLabel(backend->getPhysicalMemory())},
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
            .title = "WebGPU Backend",
            .rows = {
                {"Device", backend->getDeviceName()},
                {"API Version", backend->getApiVersion()},
                {"Physical Memory", MemoryLabel(backend->getPhysicalMemory())},
                {"Unified Memory", backend->hasUnifiedMemory() ? "Yes" : "No"},
            },
        });
    }
#endif

    return tables;
}

}  // namespace Jetstream
