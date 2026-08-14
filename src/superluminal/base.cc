#include <cmath>
#include <limits>
#include <optional>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstdint>

#include "jetstream/detail/instance_remote_supervisor.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/platform.hh"
#include "jetstream/superluminal.hh"
#include "jetstream/macros.hh"
#include "jetstream/module_surface.hh"
#include "imgui.h"

#include "dmi_block.hh"

namespace Jetstream {

namespace {

std::atomic_flag shutdownRequested = ATOMIC_FLAG_INIT;

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC) || defined(JST_OS_WINDOWS)
void HandleInterrupt() noexcept {
    constexpr char ShutdownMessage[] =
        "\n[SUPERLUMINAL] Shutdown requested. Press Ctrl+C again to force termination.\n";
    constexpr char ForceShutdownMessage[] =
        "\n[SUPERLUMINAL] Forcing immediate shutdown.\n";

    if (!shutdownRequested.test_and_set(std::memory_order_relaxed)) {
        Platform::WriteInterruptMessage(ShutdownMessage, sizeof(ShutdownMessage) - 1);
        return;
    }

    Platform::WriteInterruptMessage(ForceShutdownMessage, sizeof(ForceShutdownMessage) - 1);
    Platform::ForceTerminate(130);
}
#endif

}  // namespace

JETSTREAM_API Result PrepareSuperluminalPlotBuffer(const std::string& name,
                                                   const Superluminal::PlotConfig& config,
                                                   Superluminal::PlotConfig& resolvedConfig) {
    resolvedConfig = config;
    resolvedConfig.buffer = config.buffer.clone();

    const I32 rank = static_cast<I32>(resolvedConfig.buffer.rank());
    if (rank == 0) {
        JST_ERROR("[SUPERLUMINAL] Plot '{}' requires a non-scalar input buffer.", name);
        return Result::ERROR;
    }
    if (resolvedConfig.batchAxis < -1 || resolvedConfig.batchAxis >= rank) {
        JST_ERROR("[SUPERLUMINAL] Plot '{}' batch axis {} is out of bounds for rank {}.",
                  name, resolvedConfig.batchAxis, rank);
        return Result::ERROR;
    }
    if (resolvedConfig.channelAxis < -1 || resolvedConfig.channelAxis >= rank) {
        JST_ERROR("[SUPERLUMINAL] Plot '{}' channel axis {} is out of bounds for rank {}.",
                  name, resolvedConfig.channelAxis, rank);
        return Result::ERROR;
    }
    if ((resolvedConfig.channelAxis == -1) !=
        (resolvedConfig.channelIndex == -1)) {
        JST_ERROR("[SUPERLUMINAL] Plot '{}' requires both channel axis and channel index.",
                  name);
        return Result::ERROR;
    }
    if (resolvedConfig.channelAxis != -1 &&
        (resolvedConfig.channelIndex < 0 ||
         static_cast<U64>(resolvedConfig.channelIndex) >=
             resolvedConfig.buffer.shape(resolvedConfig.channelAxis))) {
        JST_ERROR("[SUPERLUMINAL] Plot '{}' channel index {} is out of bounds for axis {}.",
                  name, resolvedConfig.channelIndex, resolvedConfig.channelAxis);
        return Result::ERROR;
    }
    if (resolvedConfig.batchAxis != -1 &&
        resolvedConfig.batchAxis == resolvedConfig.channelAxis) {
        JST_ERROR("[SUPERLUMINAL] Plot '{}' batch and channel axes must be different.", name);
        return Result::ERROR;
    }

    SignalAxes axes;
    if (resolvedConfig.batchAxis != -1) {
        axes.batch = static_cast<Index>(resolvedConfig.batchAxis);
    }
    if (resolvedConfig.channelAxis != -1) {
        axes.channel = static_cast<Index>(resolvedConfig.channelAxis);
    }

    std::vector<Index> unclassifiedAxes;
    for (Index axis = 0; axis < resolvedConfig.buffer.rank(); ++axis) {
        if ((axes.batch && axis == *axes.batch) ||
            (axes.channel && axis == *axes.channel)) {
            continue;
        }
        unclassifiedAxes.push_back(axis);
    }
    if (unclassifiedAxes.empty()) {
        JST_ERROR("[SUPERLUMINAL] Plot '{}' has no sample axis.", name);
        return Result::ERROR;
    }

    axes.sample = unclassifiedAxes.back();
    if (!axes.batch && unclassifiedAxes.size() > 1) {
        axes.batch = unclassifiedAxes.front();
    }
    for (const Index axis : unclassifiedAxes) {
        if (axis != *axes.sample && (!axes.batch || axis != *axes.batch)) {
            JST_ERROR("[SUPERLUMINAL] Plot '{}' has an unclassified input axis {}. "
                      "Configure batchAxis and channelAxis.", name, axis);
            return Result::ERROR;
        }
    }

    return SetSignalAxes(resolvedConfig.buffer, axes);
}

struct Superluminal::Impl {
    InstanceConfig config;
    std::shared_ptr<Instance> instance;
    std::shared_ptr<Flowgraph> flowgraph;
    bool initialized;
    bool running;
    bool interruptHandlerInstalled = false;

    std::atomic_flag computeSync = ATOMIC_FLAG_INIT;

    std::thread computeThread;
    std::thread presentThread;
    std::unique_ptr<Instance::Remote::Supervisor> supervisor;

    Extent2D<U8> mosaicDims;

    struct PlotState {
        std::string name;
        Mosaic mosaic;
        Extent2D<U8> mosaicOffset;
        Extent2D<U8> mosaicSize;
        PlotConfig config;
        std::string block;
        std::function<void()> callback;
        bool active = false;
        U64 surfaceWidth = 0;
        U64 surfaceHeight = 0;
    };

    std::unordered_map<std::string, PlotState> plots;

    Result createGraph();
    Result destroyGraph();

    Result validateBounds();
    Result validateMosaic(const Mosaic& mosaic);
    Result validateName(const std::string& name);

    Result calculateMosaicParams(const Mosaic& mosaic, PlotState& state);

    Result buildLinePlotGraph(PlotState& state);
    Result buildWaterfallPlotGraph(PlotState& state);
    Result buildSpectrumAnalyzerPlotGraph(PlotState& state);
    Result buildScatterPlotGraph(PlotState& state);

    struct GraphNode {
        std::string module;
        std::string device;
        std::vector<std::string> dataType;
        std::unordered_map<std::string, std::string> config;
        std::unordered_map<std::string, std::string> input;
    };

    typedef std::vector<std::tuple<std::string, GraphNode>> Graph;

    static std::string ParseLinkDomain(const std::string& value, const std::string& domain);
    static std::vector<char> GraphToYaml(const Graph& graph, std::string domain = {});
    static U64 BufferKey(const Tensor& buffer, Domain source);
};

Superluminal::Superluminal() : impl(std::make_unique<Impl>()) {
    impl->initialized = false;
    impl->running = false;
}

Superluminal::~Superluminal() {
    if (impl->initialized) {
        terminate();
    }
}

Superluminal* Superluminal::GetInstance() {
    static Superluminal instance;
    return &instance;
}

Result Superluminal::initialize(const InstanceConfig& config) {
    JST_DEBUG("[SUPERLUMINAL] Initializing.");

    if (impl->initialized) {
        JST_CHECK(terminate());
    }

    if (config.remote && config.device != DeviceType::None && config.device != DeviceType::Vulkan) {
        JST_ERROR("[SUPERLUMINAL] Remote requires the Vulkan backend.");
        return Result::ERROR;
    }

    // Copy configuration to memory.

    impl->config = config;

    // Initialize the instance.

    Instance::Config instanceConfig = {
        .deviceId = impl->config.deviceId,
        .size = impl->config.interfaceSize,
        .scale = impl->config.interfaceScale,
    };

    if (impl->config.remote && impl->config.device == DeviceType::None) {
        instanceConfig.device = DeviceType::Vulkan;
    } else if (impl->config.device != DeviceType::None) {
        instanceConfig.device = impl->config.device;
    }

    impl->instance = std::make_shared<Instance>();
    auto result = impl->instance->create(instanceConfig);
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        impl->instance.reset();
        return result;
    }

    if (impl->config.remote) {
        Instance::Remote::Config remoteConfig;
        remoteConfig.broker = impl->config.remoteBroker;
        try {
            remoteConfig.codec = StringToRemoteCodec(impl->config.remoteCodec);
            remoteConfig.encoder = StringToRemoteEncoder(impl->config.remoteEncoder);
        } catch (const Result&) {
            JST_CHECK(impl->instance->destroy());
            impl->instance.reset();
            return Result::ERROR;
        }
        remoteConfig.autoJoinSessions = impl->config.remoteAutoJoin;
        remoteConfig.framerate = impl->config.remoteFramerate;
        result = impl->instance->remote()->create(remoteConfig);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            JST_CHECK(impl->instance->destroy());
            impl->instance.reset();
            return result;
        }
    }

    // Update the state.

    impl->config = config;
    impl->initialized = true;

    JST_INFO("[SUPERLUMINAL] Instance initialized.");
    return Result::SUCCESS;
}

Result Superluminal::terminate() {
    JST_DEBUG("[SUPERLUMINAL] Terminating.");

    // Check boundaries.

    if (!impl->initialized) {
        return Result::SUCCESS;
    }

    if (impl->running) {
        JST_CHECK(stop());
    }

    if (impl->interruptHandlerInstalled) {
        shutdownRequested.test_and_set(std::memory_order_relaxed);
    }

    if (impl->supervisor) {
        impl->supervisor->stop();
        impl->supervisor.reset();
    }

    if (impl->config.remote && impl->instance->remote()->started()) {
        JST_CHECK(impl->instance->remote()->destroy());
    }

    // Destroy instance.

    JST_CHECK(impl->instance->destroy());
    impl->instance.reset();

    // Destroy backend.

    Backend::DestroyAll();

    // Update the state.

    impl->initialized = false;
    impl->running = false;

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC) || defined(JST_OS_WINDOWS)
    if (impl->interruptHandlerInstalled) {
        Platform::UninstallInterruptHandler();
        impl->interruptHandlerInstalled = false;
    }
#endif
    shutdownRequested.clear(std::memory_order_relaxed);

    JST_INFO("[SUPERLUMINAL] Instance terminated.");
    return Result::SUCCESS;
}

Result Superluminal::start() {
    JST_DEBUG("[SUPERLUMINAL] Starting presentation.");

    // Check boundaries.

    if (!impl->initialized) {
        JST_FATAL("[SUPERLUMINAL] Instance was not initialized.");
        JST_CHECK_THROW(Result::ERROR);
    }

    if (impl->running) {
        JST_WARN("[SUPERLUMINAL] Instance is already running.");
        return Result::SUCCESS;
    }

    shutdownRequested.clear(std::memory_order_relaxed);

    // Create graph.

    JST_CHECK(impl->createGraph());

    // Start instance.

    JST_CHECK(impl->instance->start());

    // Customize ImGui style.

    ImGui::GetStyle().ScaleAllSizes(impl->config.interfaceScale);
    ImGui::GetStyle().Colors[ImGuiCol_WindowBg] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
    ImGui::GetStyle().WindowRounding = 0.0f;

    if (impl->config.remote && impl->instance->remote()) {
        impl->supervisor = std::make_unique<Instance::Remote::Supervisor>(
            impl->instance->remote().get(),
            impl->config.remoteAutoJoin);
        impl->supervisor->start();
    }

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC) || defined(JST_OS_WINDOWS)
    if (!impl->interruptHandlerInstalled) {
        impl->interruptHandlerInstalled = Platform::InstallInterruptHandler(HandleInterrupt);
        if (!impl->interruptHandlerInstalled) {
            JST_WARN("[SUPERLUMINAL] Interrupt handling is unavailable.");
        }
    }
#endif

    // Start the compute, present, and input threads.

    impl->computeSync.test_and_set();
    impl->computeThread = std::thread([&]{
        while (impl->instance->computing()) {
            impl->computeSync.wait(true);

            if (!impl->instance->computing()) {
                break;
            }

            JST_CHECK_THROW(impl->instance->compute());

            impl->computeSync.test_and_set();
            impl->computeSync.notify_all();
        }

        JST_DEBUG("[SUPERLUMINAL] Compute thread safed.");
    });

    impl->presentThread = std::thread([&]{
        while (impl->instance->presenting()) {
            auto res = impl->instance->present([&]() -> Result {
                if (!impl->running) {
                    return Result::SUCCESS;
                }

                for (auto& [_, plot] : impl->plots) {
                    static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                                    ImGuiWindowFlags_NoMove |
                                                    ImGuiWindowFlags_NoSavedSettings;

                    // TODO: Implement more than one block for mosaic.

                    const ImGuiViewport* viewport = ImGui::GetMainViewport();

                    ImVec2 workSize = {
                        (viewport->WorkSize.x / impl->mosaicDims.x) * plot.mosaicSize.x,
                        (viewport->WorkSize.y / impl->mosaicDims.y) * plot.mosaicSize.y
                    };

                    ImVec2 workPos = {
                        viewport->WorkPos.x + ((viewport->WorkSize.x / impl->mosaicDims.x) * plot.mosaicOffset.x),
                        viewport->WorkPos.y + ((viewport->WorkSize.y / impl->mosaicDims.y) * plot.mosaicOffset.y)
                    };

                    ImGui::SetNextWindowPos(workPos);
                    ImGui::SetNextWindowSize(workSize);

                    ImGui::Begin(plot.name.c_str(), nullptr, flags);

                    if (plot.config.type == Type::Interface) {
                        plot.callback();
                    } else if (!plot.block.empty()) {
                        std::vector<std::shared_ptr<Module::Surface>> surfaces;
                        if (impl->flowgraph->view().surfaces(plot.block, surfaces) == Result::SUCCESS) {
                            for (const auto& surface : surfaces) {
                                for (const auto& manifest : surface->manifests()) {
                                    const auto availableRegion = ImGui::GetContentRegionAvail();
                                    const auto& io = ImGui::GetIO();

                                    const U64 expectedWidth = availableRegion.x * io.DisplayFramebufferScale.x;
                                    const U64 expectedHeight = availableRegion.y * io.DisplayFramebufferScale.y;

                                    if (plot.surfaceWidth != static_cast<U64>(availableRegion.x) ||
                                        plot.surfaceHeight != static_cast<U64>(availableRegion.y) ||
                                        manifest.size.x != expectedWidth ||
                                        manifest.size.y != expectedHeight) {
                                        plot.surfaceWidth = availableRegion.x;
                                        plot.surfaceHeight = availableRegion.y;

                                        SurfaceEvent event;
                                        event.type = SurfaceEventType::Resize;
                                        event.size = {expectedWidth, expectedHeight};
                                        event.scale = 0.5f * impl->config.interfaceScale * io.DisplayFramebufferScale.x;
                                        event.backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};
                                        surface->pushSurfaceEvent(event);
                                    }

                                    const auto cursorPos = ImGui::GetCursorScreenPos();
                                    ImGui::Image(ImTextureRef(manifest.surface->raw()), availableRegion);

                                    ImGui::SetCursorScreenPos(cursorPos);
                                    ImGui::InvisibleButton("##surface", availableRegion);

                                    if (ImGui::IsItemHovered()) {
                                        const auto mousePos = ImGui::GetMousePos();
                                        const Extent2D<F32> normPos = {
                                            (mousePos.x - cursorPos.x) / availableRegion.x,
                                            (mousePos.y - cursorPos.y) / availableRegion.y
                                        };

                                        MouseEvent event;
                                        event.position = normPos;
                                        event.scroll = {0.0f, 0.0f};

                                        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                                            event.type = MouseEventType::Click;
                                            event.button = MouseButton::Left;
                                            surface->pushMouseEvent(event);
                                        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                                            event.type = MouseEventType::Click;
                                            event.button = MouseButton::Right;
                                            surface->pushMouseEvent(event);
                                        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                                            event.type = MouseEventType::Release;
                                            event.button = MouseButton::Left;
                                            surface->pushMouseEvent(event);
                                        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                                            event.type = MouseEventType::Release;
                                            event.button = MouseButton::Right;
                                            surface->pushMouseEvent(event);
                                        }

                                        if (io.MouseWheel != 0.0f || io.MouseWheelH != 0.0f) {
                                            event.type = MouseEventType::Scroll;
                                            event.scroll = {io.MouseWheelH, io.MouseWheel};
                                            surface->pushMouseEvent(event);
                                        }

                                        event.type = MouseEventType::Move;
                                        surface->pushMouseEvent(event);
                                    }
                                }
                            }
                        }
                    }

                    ImGui::End();
                }

                return Result::SUCCESS;
            });

            if (res != Result::SUCCESS) {
                break;
            }
        }

        JST_DEBUG("[SUPERLUMINAL] Present thread safed.");
    });

    // Update the state.

    impl->running = true;

    JST_INFO("[SUPERLUMINAL] Instance started successfully.");
    return Result::SUCCESS;
}

Result Superluminal::stop() {
    JST_DEBUG("[SUPERLUMINAL] Stopping presentation.");

    // Check boundaries.

    if (!impl->initialized) {
        JST_WARN("[SUPERLUMINAL] Can't stop because the instance is not initialized.");
        return Result::SUCCESS;
    }

    if (!impl->running) {
        JST_WARN("[SUPERLUMINAL] Can't stop because the instance is not running.");
        return Result::SUCCESS;
    }

    if (impl->interruptHandlerInstalled) {
        shutdownRequested.test_and_set(std::memory_order_relaxed);
    }

    // Update the state.

    impl->running = false;

    if (impl->supervisor) {
        impl->supervisor->stop();
        impl->supervisor.reset();
    }

    // Request to end the instance.

    impl->instance->stop();

    // Wait for the instance to end.

    impl->computeSync.clear();
    impl->computeSync.notify_all();
    if (impl->computeThread.joinable()) {
        impl->computeThread.join();
    }

    if (impl->presentThread.joinable()) {
        impl->presentThread.join();
    }

    // Destroy graph.

    JST_CHECK(impl->destroyGraph());

    shutdownRequested.clear(std::memory_order_relaxed);

    JST_INFO("[SUPERLUMINAL] Instance stopped successfully.");
    return Result::SUCCESS;
}

Result Superluminal::update(const std::string&) {
    // TODO: Implement plot level update logic.

    impl->computeSync.clear();
    impl->computeSync.notify_all();

    return Result::SUCCESS;
}

bool Superluminal::presenting() {
    return !shutdownRequested.test(std::memory_order_relaxed) && impl->instance->polling();
}

Result Superluminal::block() {
    // Check boundaries.

    if (!impl->initialized) {
        JST_WARN("[SUPERLUMINAL] Can't block because the instance is not initialized.");
        return Result::SUCCESS;
    }

    if (!impl->running) {
        JST_WARN("[SUPERLUMINAL] Can't block because the instance is not running.");
        return Result::SUCCESS;
    }

    // Block until the instance is done.

    while (!shutdownRequested.test(std::memory_order_relaxed) && impl->instance->polling()) {
        JST_CHECK(impl->instance->poll());
    }

    return Result::SUCCESS;
}

Result Superluminal::pollEvents(const bool& wait) {
    // Check boundaries.

    if (!impl->initialized) {
        return Result::SUCCESS;
    }

    if (!impl->running) {
        return Result::SUCCESS;
    }

    if (shutdownRequested.test(std::memory_order_relaxed)) {
        return Result::SUCCESS;
    }

    // Poll events.

    JST_CHECK(impl->instance->poll(wait));

    return Result::SUCCESS;
}

std::string Superluminal::RemoteRoomId() {
    auto* instance = GetInstance();
    if (!instance->impl->initialized || !instance->impl->config.remote) {
        return {};
    }

    const auto& remote = instance->impl->instance->remote();
    if (!remote || !remote->started()) {
        return {};
    }

    return remote->roomId();
}

std::string Superluminal::RemoteInviteUrl() {
    auto* instance = GetInstance();
    if (!instance->impl->initialized || !instance->impl->config.remote) {
        return {};
    }

    const auto& remote = instance->impl->instance->remote();
    if (!remote || !remote->started()) {
        return {};
    }

    return remote->inviteUrl();
}

std::string Superluminal::RemoteAccessToken() {
    auto* instance = GetInstance();
    if (!instance->impl->initialized || !instance->impl->config.remote) {
        return {};
    }

    const auto& remote = instance->impl->instance->remote();
    if (!remote || !remote->started()) {
        return {};
    }

    return remote->accessToken();
}

Result Superluminal::PrintRemoteInfo() {
    auto* instance = GetInstance();
    if (!instance->impl->initialized || !instance->impl->config.remote) {
        JST_WARN("[SUPERLUMINAL] Remote is not enabled.");
        return Result::SUCCESS;
    }

    const auto& remote = instance->impl->instance->remote();
    if (!remote || !remote->started()) {
        JST_WARN("[SUPERLUMINAL] Remote session is not started.");
        return Result::SUCCESS;
    }

    Instance::Remote::Supervisor supervisor(remote.get(), false);
    supervisor.print();
    return Result::SUCCESS;
}

Result Superluminal::Impl::validateMosaic(const Mosaic& mosaic) {
    // Validate mosaic size.

    Extent2D<U8> dims;
    dims.y = mosaic.size();

    if (dims.y == 0) {
        JST_FATAL("[SUPERLUMINAL] Mosaic should be a 2D matrix. Currently: '{}'. Example: '{{0}}', or '{{{{0, 0}}, {{0, 1}}}}'.", mosaic);
        return Result::ERROR;
    }

    dims.x = mosaic[0].size();
    for (auto& column : mosaic) {
        if (dims.x != column.size()) {
            JST_FATAL("[SUPERLUMINAL] All mosaic rows should have the same size");
            return Result::ERROR;
        }
    }

    if (plots.size() == 0) {
        mosaicDims = dims;
    } else {
        if (mosaicDims != dims) {
            JST_FATAL("[SUPERLUMINAL] The mosaic dimensions of all plots need to be the same.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result Superluminal::Impl::validateName(const std::string& name) {
    // Check plot name.

    if (name.empty()) {
        JST_FATAL("[SUPERLUMINAL] Plot name cannot be empty.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Superluminal::Impl::calculateMosaicParams(const Mosaic& mosaic, PlotState& state) {
    // Calculate mosaic offset.

    state.mosaicOffset = [&](){
        for (U8 x = 0; x < mosaicDims.x; x++) {
            for (U8 y = 0; y < mosaicDims.y; y++) {
                if (mosaic[y][x] != 0) {
                    return Extent2D<U8>{x, y};
                }
            }
        }
        return Extent2D<U8>{0, 0};
    }();

    JST_DEBUG("[SUPERLUMINAL] Mosaic offsets for plot '{}' is (X: {}, Y: {}).", state.name,
                                                                                state.mosaicOffset.x,
                                                                                state.mosaicOffset.y);

    // Calculate mosaic size.

    state.mosaicSize = [&](){
        U8 x = 0;
        U8 y = 0;

        for (U8 i = state.mosaicOffset.x; i < mosaicDims.x; i++) {
            if (mosaic[state.mosaicOffset.y][i] != 0) {
                x += 1;
            }
        }

        for (U8 i = state.mosaicOffset.y; i < mosaicDims.y; i++) {
            if (mosaic[i][state.mosaicOffset.x] != 0) {
                y += 1;
            }
        }

        return Extent2D<U8>{x, y};
    }();

    JST_DEBUG("[SUPERLUMINAL] Mosaic size for plot '{}' is (X: {}, Y: {}).", state.name,
                                                                             state.mosaicSize.x,
                                                                             state.mosaicSize.y);

    return Result::SUCCESS;
}

Result Superluminal::interface(const std::string& name, const Mosaic& mosaic, const std::function<void()>& callback) {
    JST_DEBUG("[SUPERLUMINAL] Registering new interface called '{}'.", name);

    // Check boundaries.

    if (!impl->initialized) {
        JST_CHECK(initialize());
    }

    if (impl->running) {
        JST_FATAL("[SUPERLUMINAL] Can't register new interface because the instance is already commited.");
        return Result::ERROR;
    }

    JST_CHECK(impl->validateMosaic(mosaic));
    JST_CHECK(impl->validateName(name));

    // Create plot state.

    auto& state = impl->plots[name];

    state.config.type = Type::Interface;
    state.callback = callback;
    state.mosaic = mosaic;
    state.name = name;

    JST_CHECK(impl->calculateMosaicParams(mosaic, state));

    JST_INFO("[SUPERLUMINAL] Created interface '{}'.", state.name);
    return Result::SUCCESS;
}

Result Superluminal::plot(const std::string& name, const Mosaic& mosaic, const PlotConfig& config) {
    JST_DEBUG("[SUPERLUMINAL] Registering new plot called '{}'.", name);

    // Check boundaries.

    if (!impl->initialized) {
        JST_CHECK(initialize());
    }

    if (impl->running) {
        JST_FATAL("[SUPERLUMINAL] Can't register new plot because the instance is already commited.");
        return Result::ERROR;
    }

    JST_CHECK(impl->validateMosaic(mosaic));
    JST_CHECK(impl->validateName(name));

    PlotConfig resolvedConfig;
    JST_CHECK(PrepareSuperluminalPlotBuffer(name, config, resolvedConfig));

    F32 frequency = 0.0f;
    F32 sampleRate = 0.0f;
    bool hasFrequency = false;
    bool hasSampleRate = false;

    const auto readMetadataOption = [&](const std::string& key,
                                        F32& value,
                                        bool& present) -> Result {
        const auto it = config.options.find(key);
        if (it == config.options.end()) {
            return Result::SUCCESS;
        }

        present = true;
        if (const auto* number = std::get_if<F32>(&it->second)) {
            value = *number;
        } else if (const auto* integer = std::get_if<I32>(&it->second)) {
            value = static_cast<F32>(*integer);
        } else {
            JST_ERROR("[SUPERLUMINAL] Plot option '{}' must be numeric.", key);
            return Result::ERROR;
        }

        if (!std::isfinite(value)) {
            JST_ERROR("[SUPERLUMINAL] Plot option '{}' must be finite.", key);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    };

    JST_CHECK(readMetadataOption("frequency", frequency, hasFrequency));
    JST_CHECK(readMetadataOption("sampleRate", sampleRate, hasSampleRate));

    if (hasSampleRate && sampleRate <= 0.0f) {
        JST_ERROR("[SUPERLUMINAL] Plot option 'sampleRate' must be positive.");
        return Result::ERROR;
    }

    if (hasFrequency) {
        JST_CHECK(resolvedConfig.buffer.setAttribute("frequency", frequency));
    }
    if (hasSampleRate) {
        JST_CHECK(resolvedConfig.buffer.setAttribute("sampleRate", sampleRate));
    }

    // Create plot state.

    auto& state = impl->plots[name];

    state.config = std::move(resolvedConfig);
    state.mosaic = mosaic;
    state.name = name;

    JST_CHECK(impl->calculateMosaicParams(mosaic, state));

    JST_INFO("[SUPERLUMINAL] Created plot '{}'.", state.name);
    return Result::SUCCESS;
}

std::vector<std::vector<U8>> Superluminal::MosaicLayout(U8 matrixHeight, U8 matrixWidth,
                                                        U8 panelHeight, U8 panelWidth,
                                                        U8 offsetX, U8 offsetY) {
    std::vector<std::vector<U8>> layout(matrixHeight, std::vector<U8>(matrixWidth, 0));

    for (int i = 0; i < panelHeight; ++i) {
        for (int j = 0; j < panelWidth; ++j) {
            int row = offsetY + i;
            int col = offsetX + j;
            if (row < matrixHeight && col < matrixWidth) {
                layout[row][col] = 1;
            }
        }
    }

    return layout;
}

Result Superluminal::Impl::createGraph() {
    JST_DEBUG("[SUPERLUMINAL] Create graph.");

    // Create flowgraph.

    JST_CHECK(instance->flowgraphCreate("superluminal", {}, flowgraph));

    // Import memory buffers.

    struct InputMemoryRecipe {
        Tensor buffer;
        Domain source;
        std::unordered_set<Domain> display;
    };

    std::unordered_map<U64, InputMemoryRecipe> buffer_map;

    for (auto& [_, state] : plots) {
        if (state.config.type == Type::Interface) {
            continue;
        }

        const auto& buf = state.config.buffer;
        auto& recipe = buffer_map[BufferKey(buf, state.config.source)];

        recipe.buffer = buf;
        recipe.source = state.config.source;
        recipe.display.insert(state.config.type == Type::SpectrumAnalyzer
            ? state.config.source
            : state.config.display);
    }

    for (auto& [hash, recipe] : buffer_map) {
        auto sourceDomain = (recipe.source == Domain::Time) ? "time" : "freq";
        auto conversionDomain = (recipe.source != Domain::Time) ? "time" : "freq";

        // Create DMI block for this buffer.

        auto blockName = jst::fmt::format("data_{}_{}_{}", GetDeviceName(recipe.buffer.device()), sourceDomain, hash);

        Blocks::DynamicTensorImport dtiConfig;
        dtiConfig.buffer = recipe.buffer;
        JST_CHECK(flowgraph->blockCreate(blockName, dtiConfig, {}, recipe.buffer.device()));

        // Handle device transfer if needed.

        if (recipe.buffer.device() != config.preferredDevice) {
            std::string deviceNameStr;

            if ((recipe.buffer.device() == DeviceType::CUDA) && (config.preferredDevice == DeviceType::CPU)) {
                deviceNameStr = GetDeviceName(recipe.buffer.device());
            }

            if ((recipe.buffer.device() == DeviceType::CPU) && (config.preferredDevice == DeviceType::CUDA)) {
                deviceNameStr = GetDeviceName(recipe.buffer.device());
            }

            if (deviceNameStr.empty()) {
                JST_ERROR("[SUPERLUMINAL] Unsupported device conversion.");
                return Result::ERROR;
            }

            auto dtypeName = DataTypeToName(recipe.buffer.dtype());
            auto blob = GraphToYaml({
                {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), sourceDomain, hash),
                    {"duplicate", deviceNameStr, {std::string(dtypeName)}, {
                        {{"hostAccessible", "true"},
                         {"outputDevice", GetDeviceName(config.preferredDevice)}}},
                        {{"buffer", jst::fmt::format("${{graph.{}.output.buffer}}", blockName)}}}},
            });

            JST_CHECK(flowgraph->importFromBlob(blob));
        }

        // Check if a conversion is needed.

        if (recipe.display.contains(recipe.source) && recipe.display.size() == 1) {
            JST_DEBUG("[SUPERLUMINAL] Skipping conversion for '{}'.", hash);
            continue;
        }

        // Fulfills Time -> Frequency and Frequency -> Time conversions.

        auto forward = (recipe.source == Domain::Time && recipe.display.contains(Domain::Frequency));
        bool isComplexBuffer = (recipe.buffer.dtype() == DataType::CF32);

        if (isComplexBuffer) {
            // Complex signal path - needs windowing, inversion, and multiplication
            if (config.preferredDevice == DeviceType::CPU) {
                SignalAxes axes;
                JST_CHECK(ResolveSignalAxes(recipe.buffer, axes));
                auto winName = jst::fmt::format("win_{}", hash);
                auto invName = jst::fmt::format("inv_{}", hash);
                auto reshapeName = jst::fmt::format("win_shape_{}", hash);
                auto mulName = jst::fmt::format("win_mul_{}", hash);
                std::string windowShape = "[";
                for (Index axis = 0; axis < recipe.buffer.rank(); ++axis) {
                    if (axis > 0) {
                        windowShape += ", ";
                    }
                    windowShape += std::to_string(
                        axis == *axes.sample
                            ? recipe.buffer.shape(*axes.sample)
                            : 1);
                }
                windowShape += "]";
                const auto windowSize =
                    std::to_string(recipe.buffer.shape(*axes.sample));

                auto blob = GraphToYaml({
                    {winName,
                        {"window", GetDeviceName(config.preferredDevice), {"CF32"},
                            {{"size", windowSize}}, {}}},
                    {invName,
                        {"invert", GetDeviceName(config.preferredDevice), {"CF32"}, {},
                            {{"signal", jst::fmt::format("${{graph.{}.output.window}}", winName)}}}},
                    {reshapeName,
                        {"reshape", GetDeviceName(config.preferredDevice), {"CF32"},
                            {{"shape", windowShape}},
                            {{"buffer", jst::fmt::format(
                                "${{graph.{}.output.signal}}", invName)}}}},
                    {mulName,
                        {"multiply", GetDeviceName(config.preferredDevice), {"CF32"}, {},
                            {{"a", jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), sourceDomain, hash)},
                            {"b", jst::fmt::format(
                                "${{graph.{}.output.buffer}}", reshapeName)}}}},
                    {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), conversionDomain, hash),
                        {"fft", GetDeviceName(config.preferredDevice), {"CF32", "CF32"},
                            {{"forward", jst::fmt::format("{}", (forward) ? "true" : "false")}},
                            {{"signal", jst::fmt::format("${{graph.{}.output.product}}", mulName)}}}},
                });
                JST_CHECK(flowgraph->importFromBlob(blob));
            } else {
                // TODO: The Multiply block doesn't support CUDA yet. This is a temporary bypass.
                auto blob = GraphToYaml({
                    {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), conversionDomain, hash),
                        {"fft", GetDeviceName(config.preferredDevice), {"CF32", "CF32"},
                            {{"forward", jst::fmt::format("{}", (forward) ? "true" : "false")}},
                            {{"signal",  jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), sourceDomain, hash)}}}},
                });
                JST_CHECK(flowgraph->importFromBlob(blob));
            }
        } else {
            // Real signal path - direct FFT without windowing for domain conversion
            auto blob = GraphToYaml({
                {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), conversionDomain, hash),
                    {"fft", GetDeviceName(config.preferredDevice), {"F32", "F32"},
                        {{"forward", jst::fmt::format("{}", (forward) ? "true" : "false")}},
                        {{"signal",  jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), sourceDomain, hash)}}}},
            });
            JST_CHECK(flowgraph->importFromBlob(blob));
        }
    }

    // Create plots graph.

    for (auto& [name, state] : plots) {
        switch (state.config.type) {
            case Type::Line:
                JST_CHECK(buildLinePlotGraph(state));
                break;
            case Type::Waterfall:
                JST_CHECK(buildWaterfallPlotGraph(state));
                break;
            case Type::SpectrumAnalyzer:
                JST_CHECK(buildSpectrumAnalyzerPlotGraph(state));
                break;
            case Type::Interface:
                break;
            case Type::Scatter:
                JST_CHECK(buildScatterPlotGraph(state));
                break;
            case Type::Heat:
                JST_FATAL("[SUPERLUMINAL] Plot type for '{}' not implemented yet.", name);
                break;
        }
    }

    return Result::SUCCESS;
}

Result Superluminal::Impl::destroyGraph() {
    JST_DEBUG("[SUPERLUMINAL] Destroy graph.");

    // Destroy plots graph.

    for (auto& [_, state] : plots) {
        state.block = {};
    }

    // Destroy flowgraph.

    JST_CHECK(instance->flowgraphDestroy("superluminal"));
    flowgraph.reset();

    return Result::SUCCESS;
}

Result Superluminal::Impl::buildLinePlotGraph(PlotState& state) {
    JST_DEBUG("[SUPERLUMINAL] Building line plot graph named '{}'.", state.name);

    // Access buffer metadata.

    const auto& buf = state.config.buffer;

    if (state.config.options.contains("waterfall") ||
        state.config.options.contains("waterfallHeight") ||
        state.config.options.contains("waterfallInterpolate")) {
        JST_ERROR("[SUPERLUMINAL] Integrated Line waterfall options are no "
                  "longer supported; use SpectrumAnalyzer or a separate "
                  "Waterfall plot.");
        return Result::ERROR;
    }

    // Poll options.

    std::string averagingRate = "1";

    if (state.config.options.contains("averaging")) {
        auto averaging = std::get<I32>(state.config.options["averaging"]);
        JST_DEBUG("[SUPERLUMINAL] Averaging set to {}.", averaging);
        averagingRate = std::to_string(averaging);
    }

    std::string decimationRate = "1";

    if (state.config.options.contains("decimation")) {
        auto decimation = std::get<I32>(state.config.options["decimation"]);
        JST_DEBUG("[SUPERLUMINAL] Decimation set to {}.", decimation);
        decimationRate = std::to_string(decimation);
    }

    std::unordered_map<std::string, std::string> lineplotConfig = {
        {"averaging", averagingRate},
        {"decimation", decimationRate},
    };
    bool fillEnabled = state.config.display == Domain::Frequency;
    if (state.config.options.contains("fill")) {
        const auto& value = state.config.options.at("fill");
        if (const auto* integer = std::get_if<I32>(&value)) {
            fillEnabled = *integer != 0;
        } else if (const auto* number = std::get_if<F32>(&value)) {
            fillEnabled = *number != 0.0f;
        } else {
            JST_ERROR("[SUPERLUMINAL] Plot option 'fill' must be boolean.");
            return Result::ERROR;
        }
    }
    lineplotConfig["fill"] = fillEnabled ? "true" : "false";
    const bool timeDomainReal = state.config.buffer.dtype() == DataType::F32 &&
                                state.config.display == Domain::Time;
    lineplotConfig["rangeMin"] = timeDomainReal ? "-1" : "-100";
    lineplotConfig["rangeMax"] = timeDomainReal ? "1" : "0";
    for (const std::string& key : {"xLabel", "yLabel"}) {
        if (state.config.options.contains(key)) {
            lineplotConfig[key] =
                std::get<std::string>(state.config.options.at(key));
        }
    }

    // Build graph.

    auto domain = (state.config.display == Domain::Time) ? "time" : "freq";
    auto hash = std::to_string(BufferKey(buf, state.config.source));
    auto outputPort = (state.config.display == state.config.source) ? "buffer" : "signal";
    auto port = jst::fmt::format("${{graph.data_{}_{}_{}.output.{}}}", GetDeviceName(config.preferredDevice), domain, hash, outputPort);

    bool isRealBuffer = (buf.dtype() == DataType::F32);
    const std::string inputType = isRealBuffer ? "F32" : "CF32";
    auto graph = Graph{};

    if (state.config.channelAxis != -1) {
        const U64 channelAxis = state.config.channelAxis;
        const U64 channelIndex = state.config.channelIndex;
        std::string slice;
        for (U64 axis = 0; axis < buf.rank(); ++axis) {
            slice += axis == channelAxis ? std::to_string(channelIndex) : ":";
            if (axis + 1 != buf.rank()) {
                slice += ",";
            }
        }
        slice = jst::fmt::format("[{}]", slice);

        graph.push_back({
            "slice",
            {"slice", GetDeviceName(config.preferredDevice), {inputType},
                {{"slice", slice}},
                {{"buffer", port}}},
        });
        graph.push_back({
            "duplicate",
            {"duplicate", GetDeviceName(config.preferredDevice), {inputType},
                {{"outputDevice", GetDeviceName(config.preferredDevice)}},
                {{"buffer", "${domain.slice.output.buffer}"}}},
        });
        port = "${domain.duplicate.output.buffer}";
    }

    // For real signals in time domain, we don't need amplitude conversion
    if (isRealBuffer && state.config.display == Domain::Time) {
        graph.insert(graph.end(), {
            {"scl",
                {"range", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"min", "-1"}, {"max", "1"}},
                    {{"signal", port}}}},
            {"lineplot",
                {"lineplot", GetDeviceName(config.preferredDevice), {"F32"},
                    lineplotConfig,
                    {{"signal", "${domain.scl.output.signal}"}}}},
        });
    } else {
        graph.insert(graph.end(), {
            {"amp",
                {"amplitude", GetDeviceName(config.preferredDevice),
                    {inputType, "F32"}, {},
                    {{"signal", port}}}},
            {"scl",
                {"range", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"min", "-100"}, {"max", "0"}},
                    {{"signal", "${domain.amp.output.signal}"}}}},
            {"lineplot",
                {"lineplot", GetDeviceName(config.preferredDevice), {"F32"},
                    lineplotConfig,
                    {{"signal", "${domain.scl.output.signal}"}}}},
        });
    }
    JST_CHECK(flowgraph->importFromBlob(GraphToYaml(graph, state.name)));

    // Update plot state.

    state.block = state.name + "_lineplot";

    return Result::SUCCESS;
}

Result Superluminal::Impl::buildSpectrumAnalyzerPlotGraph(PlotState& state) {
    JST_DEBUG("[SUPERLUMINAL] Building spectrum analyzer graph named '{}'.",
              state.name);

    const auto& buf = state.config.buffer;
    if (buf.dtype() != DataType::CF32) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' requires a CF32 input buffer.",
                  state.name);
        return Result::ERROR;
    }
    if (state.config.source != Domain::Time) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' requires time-domain input.",
                  state.name);
        return Result::ERROR;
    }
    if ((state.config.channelAxis == -1) != (state.config.channelIndex == -1)) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' requires both channel "
                  "axis and channel index.",
                  state.name);
        return Result::ERROR;
    }

    const I32 inputRank = static_cast<I32>(buf.rank());
    if (inputRank == 0) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' requires a non-scalar "
                  "input buffer.",
                  state.name);
        return Result::ERROR;
    }
    if (state.config.channelAxis < -1 || state.config.channelAxis >= inputRank) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' channel axis {} is "
                  "out of bounds for rank {}.",
                  state.name, state.config.channelAxis, inputRank);
        return Result::ERROR;
    }
    if (state.config.channelAxis != -1 &&
        (state.config.channelIndex < 0 ||
         static_cast<U64>(state.config.channelIndex) >=
             buf.shape()[state.config.channelAxis])) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' channel index {} is "
                  "out of bounds for axis {}.",
                  state.name, state.config.channelIndex, state.config.channelAxis);
        return Result::ERROR;
    }
    if (state.config.batchAxis < -1 || state.config.batchAxis >= inputRank) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' batch axis {} is out "
                  "of bounds for rank {}.",
                  state.name, state.config.batchAxis, inputRank);
        return Result::ERROR;
    }
    if (state.config.batchAxis != -1 &&
        state.config.batchAxis == state.config.channelAxis) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' batch and channel "
                  "axes must be different.",
                  state.name);
        return Result::ERROR;
    }

    const auto integerOption = [&](const std::string& key,
                                   const I32 fallback) -> std::optional<I32> {
        if (!state.config.options.contains(key)) { return fallback; }
        const auto& value = state.config.options.at(key);
        if (const auto* integer = std::get_if<I32>(&value)) { return *integer; }
        if (const auto* number = std::get_if<F32>(&value)) {
            const double resolved = static_cast<double>(*number);
            if (!std::isfinite(resolved) ||
                resolved < static_cast<double>(std::numeric_limits<I32>::min()) ||
                resolved > static_cast<double>(std::numeric_limits<I32>::max())) {
                JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer option '{}' must be "
                          "a finite integer within the I32 range.", key);
                return std::nullopt;
            }
            const I32 converted = static_cast<I32>(resolved);
            if (static_cast<double>(converted) != resolved) {
                JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer option '{}' must be "
                          "an integer value.", key);
                return std::nullopt;
            }
            return converted;
        }
        return fallback;
    };
    const auto floatOption = [&](const std::string& key, const F32 fallback) {
        if (!state.config.options.contains(key)) { return fallback; }
        const auto& value = state.config.options.at(key);
        if (const auto* number = std::get_if<F32>(&value)) { return *number; }
        if (const auto* integer = std::get_if<I32>(&value)) {
            return static_cast<F32>(*integer);
        }
        return fallback;
    };

    const std::string rangeMin = std::to_string(floatOption("rangeMin", -100.0f));
    const std::string rangeMax = std::to_string(floatOption("rangeMax", 0.0f));

    const auto averagingOpt = integerOption("averaging", 1);
    const auto decimationOpt = integerOption("decimation", 1);
    const auto maxHoldOpt = integerOption("maxHold", 0);
    const auto waterfallHeightOpt = integerOption("waterfallHeight", 512);
    const auto fillOpt = integerOption("fill", 1);
    if (!averagingOpt || !decimationOpt || !maxHoldOpt ||
        !waterfallHeightOpt || !fillOpt) {
        return Result::ERROR;
    }
    const std::string averaging = std::to_string(*averagingOpt);
    const std::string decimation = std::to_string(*decimationOpt);
    const std::string maxHold = *maxHoldOpt != 0 ? "true" : "false";
    const std::string waterfallHeight = std::to_string(*waterfallHeightOpt);
    const std::string fill = *fillOpt != 0 ? "true" : "false";

    auto graph = Graph{};
    const auto hash = std::to_string(BufferKey(buf, state.config.source));
    auto port = jst::fmt::format(
        "${{graph.data_{}_time_{}.output.buffer}}",
        GetDeviceName(config.preferredDevice),
        hash);
    U64 effectiveRank = buf.rank();

    if (state.config.channelAxis != -1) {
        const U64 channelAxis = state.config.channelAxis;
        const U64 channelIndex = state.config.channelIndex;
        std::string slice;
        for (U64 axis = 0; axis < buf.rank(); ++axis) {
            slice += axis == channelAxis ? std::to_string(channelIndex) : ":";
            if (axis + 1 != buf.rank()) { slice += ","; }
        }
        slice = jst::fmt::format("[{}]", slice);

        graph.push_back({
            "slice",
            {"slice", GetDeviceName(config.preferredDevice), {"CF32"},
                {{"slice", slice}},
                {{"buffer", port}}},
        });
        graph.push_back({
            "duplicate",
            {"duplicate", GetDeviceName(config.preferredDevice), {"CF32"},
                {{"outputDevice", GetDeviceName(config.preferredDevice)}},
                {{"buffer", "${domain.slice.output.buffer}"}}},
        });
        port = "${domain.duplicate.output.buffer}";
        effectiveRank -= 1;
    }

    if (effectiveRank == 0) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' has no transform axis "
                  "after channel selection.",
                  state.name);
        return Result::ERROR;
    }
    if (effectiveRank > 2) {
        JST_ERROR("[SUPERLUMINAL] Spectrum Analyzer '{}' supports at most one "
                  "batch axis and one transform axis after channel selection.",
                  state.name);
        return Result::ERROR;
    }
    std::unordered_map<std::string, std::string> spectrumAnalyzerConfig = {
        {"rangeMin", rangeMin},
        {"rangeMax", rangeMax},
        {"averaging", averaging},
        {"decimation", decimation},
        {"maxHold", maxHold},
        {"fill", fill},
        {"waterfallHeight", waterfallHeight},
    };
    for (const std::string& key : {"xLabel", "amplitudeLabel", "waterfallLabel"}) {
        if (state.config.options.contains(key)) {
            spectrumAnalyzerConfig[key] =
                std::get<std::string>(state.config.options.at(key));
        }
    }

    graph.push_back({
        "spectrum_analyzer",
        {"spectrum_analyzer", GetDeviceName(config.preferredDevice), {"CF32"},
            spectrumAnalyzerConfig,
            {{"buffer", port}}},
    });
    JST_CHECK(flowgraph->importFromBlob(GraphToYaml(graph, state.name)));

    state.block = state.name + "_spectrum_analyzer";
    return Result::SUCCESS;
}

Result Superluminal::Impl::buildWaterfallPlotGraph(PlotState& state) {
    JST_DEBUG("[SUPERLUMINAL] Building waterfall plot graph named '{}'.", state.name);

    // Access buffer metadata.

    const auto& buf = state.config.buffer;

    // Poll options.

    std::string height = "512";

    if (state.config.options.contains("height")) {
        auto h = std::get<I32>(state.config.options["height"]);
        JST_DEBUG("[SUPERLUMINAL] Height set to {}.", h);
        height = std::to_string(h);
    }

    std::unordered_map<std::string, std::string> waterfallConfig = {
        {"height", height},
    };
    for (const std::string& key : {"xLabel", "yLabel"}) {
        if (state.config.options.contains(key)) {
            waterfallConfig[key] =
                std::get<std::string>(state.config.options.at(key));
        }
    }

    // Build graph.

    auto graph = Graph{};
    auto hash = std::to_string(BufferKey(buf, state.config.source));
    auto domain = (state.config.display == Domain::Time) ? "time" : "freq";
    auto outputPort = (state.config.display == state.config.source) ? "buffer" : "signal";
    auto port = jst::fmt::format("${{graph.data_{}_{}_{}.output.{}}}", GetDeviceName(config.preferredDevice), domain, hash, outputPort);

    if (state.config.channelAxis != -1 && state.config.channelIndex != -1) {
        U64 axis = state.config.channelAxis;
        U64 index = state.config.channelIndex;

        // Parse slice string.

        std::string slice;
        for (U64 i = 0; i < buf.rank(); i++) {
            if (i == axis) {
                slice += jst::fmt::format("{}", index);
            } else {
                slice += jst::fmt::format(":");
            }
            if (i != buf.rank() - 1) {
                slice += ",";
            }
        }
        slice = jst::fmt::format("[{}]", slice);

        // Create slice module.

        graph.push_back({
            "slice",
            {"slice", GetDeviceName(config.preferredDevice),
                {std::string(DataTypeToName(buf.dtype()))},
                {{"slice", slice}},
                {{"buffer", port}}},
        });

        graph.push_back({
            "duplicate",
            {"duplicate", GetDeviceName(config.preferredDevice),
                {std::string(DataTypeToName(buf.dtype()))},
                {{"outputDevice", GetDeviceName(config.preferredDevice)}},
                {{"buffer", "${domain.slice.output.buffer}"}}},
        });

        port = jst::fmt::format("${{domain.duplicate.output.buffer}}");
    }

    bool isRealBuffer = (buf.dtype() == DataType::F32);

    // For real signals in time domain, skip amplitude conversion
    if (isRealBuffer && state.config.display == Domain::Time) {
        graph.push_back({
            "scl",
            {"range", GetDeviceName(config.preferredDevice), {"F32"},
                {{"min", "-1"}, {"max", "1"}},
                {{"signal", port}}},
        });
    } else {
        std::string inputType = isRealBuffer ? "F32" : "CF32";

        graph.push_back({
            "amp",
            {"amplitude", GetDeviceName(config.preferredDevice), {inputType, "F32"}, {},
                {{"signal", port}}},
        });

        graph.push_back({
            "scl",
            {"range", GetDeviceName(config.preferredDevice), {"F32"},
                {{"min", "-100"}, {"max", "0"}},
                {{"signal", "${domain.amp.output.signal}"}}},
        });
    }

    graph.push_back({
        "waterfall",
        {"waterfall", GetDeviceName(config.preferredDevice), {"F32"},
            waterfallConfig,
            {{"signal", "${domain.scl.output.signal}"}}},
    });

    JST_CHECK(flowgraph->importFromBlob(GraphToYaml(graph, state.name)));

    // Update plot state.

    state.block = state.name + "_waterfall";

    return Result::SUCCESS;
}

Result Superluminal::Impl::buildScatterPlotGraph(PlotState& state) {
    JST_DEBUG("[SUPERLUMINAL] Building scatter plot graph named '{}'.", state.name);

    // Access buffer metadata.

    const auto& buf = state.config.buffer;

    if (buf.dtype() != DataType::CF32) {
        JST_ERROR("[SUPERLUMINAL] Scatter plot requires a complex (CF32) buffer.");
        return Result::ERROR;
    }

    // Build graph.

    auto graph = Graph{};
    auto hash = std::to_string(BufferKey(buf, state.config.source));
    auto domain = (state.config.display == Domain::Time) ? "time" : "freq";
    auto outputPort = (state.config.display == state.config.source) ? "buffer" : "signal";
    auto port = jst::fmt::format("${{graph.data_{}_{}_{}.output.{}}}", GetDeviceName(config.preferredDevice), domain, hash, outputPort);

    std::unordered_map<std::string, std::string> constellationConfig;
    for (const std::string& key : {"xLabel", "yLabel"}) {
        if (state.config.options.contains(key)) {
            constellationConfig[key] =
                std::get<std::string>(state.config.options.at(key));
        }
    }

    if (state.config.channelAxis != -1 && state.config.channelIndex != -1) {
        U64 axis = state.config.channelAxis;
        U64 index = state.config.channelIndex;

        // Parse slice string.

        std::string slice;
        for (U64 i = 0; i < buf.rank(); i++) {
            if (i == axis) {
                slice += jst::fmt::format("{}", index);
            } else {
                slice += jst::fmt::format(":");
            }
            if (i != buf.rank() - 1) {
                slice += ",";
            }
        }
        slice = jst::fmt::format("[{}]", slice);

        // Create slice module.

        graph.push_back({
            "slice",
            {"slice", GetDeviceName(config.preferredDevice), {"CF32"},
                {{"slice", slice}},
                {{"buffer", port}}},
        });

        graph.push_back({
            "duplicate",
            {"duplicate", GetDeviceName(config.preferredDevice), {"CF32"},
                {{"outputDevice", GetDeviceName(config.preferredDevice)}},
                {{"buffer", "${domain.slice.output.buffer}"}}},
        });

        port = jst::fmt::format("${{domain.duplicate.output.buffer}}");
    }

    graph.push_back({
        "constellation",
        {"constellation", GetDeviceName(config.preferredDevice), {"CF32"},
            constellationConfig,
            {{"signal", port}}},
    });

    JST_CHECK(flowgraph->importFromBlob(GraphToYaml(graph, state.name)));

    // Update plot state.

    state.block = state.name + "_constellation";

    return Result::SUCCESS;
}

std::string Superluminal::Impl::ParseLinkDomain(const std::string& value, const std::string& domain) {
    std::regex pattern(R"(\$\{domain\.([\w\-]+)\.([\w\-]+)\.([\w\-]+)\})");
    std::smatch matches;
    if (std::regex_match(value, matches, pattern)) {
        return jst::fmt::format("${{graph.{}{}.{}.{}}}", domain, matches[1].str(), matches[2].str(), matches[3].str());
    }
    return value;
}

U64 Superluminal::Impl::BufferKey(const Tensor& buffer, const Domain source) {
    auto hashCombine = [](U64& seed, const U64 value) {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };

    U64 seed = 0;
    try {
        const auto ptr = static_cast<U64>(reinterpret_cast<std::uintptr_t>(buffer.data()));
        hashCombine(seed, ptr);
    } catch (...) {
        // Some tensor backends may not expose host-readable pointers.
        hashCombine(seed, static_cast<U64>(buffer.id()));
    }

    hashCombine(seed, static_cast<U64>(buffer.offset()));
    hashCombine(seed, static_cast<U64>(buffer.device()));
    hashCombine(seed, static_cast<U64>(buffer.dtype()));
    hashCombine(seed, static_cast<U64>(source));
    hashCombine(seed, static_cast<U64>(buffer.rank()));
    for (U64 i = 0; i < buffer.rank(); ++i) {
        hashCombine(seed, buffer.shape(i));
        hashCombine(seed, buffer.stride(i));
    }

    for (const std::string& key : {"frequency", "sampleRate"}) {
        if (!buffer.hasAttribute(key)) {
            continue;
        }

        const auto attribute = buffer.attribute(key);
        if (attribute.type() == typeid(F32)) {
            hashCombine(seed, std::hash<std::string>{}(key));
            hashCombine(seed, std::hash<F32>{}(std::any_cast<F32>(attribute)));
        }
    }

    for (const std::string_view attribute : {
             SampleAxisAttribute, BatchAxisAttribute, ChannelAxisAttribute}) {
        if (buffer.hasAttribute(std::string(attribute))) {
            hashCombine(seed, std::hash<std::string_view>{}(attribute));
            hashCombine(seed, std::any_cast<Index>(
                buffer.attribute(std::string(attribute))));
        }
    }

    if (seed == 0) {
        return static_cast<U64>(buffer.id());
    }

    return seed;
}

// TODO: Upstream to `Flowgraph` class.
std::vector<char> Superluminal::Impl::GraphToYaml(const Graph& graph, std::string domain) {
    std::string yaml = jst::fmt::format(
        "---\n"
        "version: 1.0.0\n"
        "\n"
        "graph:\n"
    );

    const auto escapeSingleQuotedScalar = [](const std::string& value) {
        std::string escaped;
        escaped.reserve(value.size());
        for (const char character : value) {
            escaped += character;
            if (character == '\'') {
                escaped += character;
            }
        }
        return escaped;
    };

    if (!domain.empty()) {
        domain += "_";
    }

    for (const auto& [key, value] : graph) {
        yaml += jst::fmt::format(
            " {}{}:\n"
            "  module: {}\n"
            "  device: {}\n",
            domain, key, value.module, value.device
        );

        if (value.dataType.size() == 1) {
            yaml += jst::fmt::format(
                "  dataType: {}\n",
                value.dataType[0]
            );
        } else {
            yaml += jst::fmt::format(
                "  inputDataType: {}\n"
                "  outputDataType: {}\n",
                value.dataType[0], value.dataType[1]
            );
        }

        if (!value.config.empty()) {
            yaml += "  config:\n";
            for (const auto& [configKey, configValue] : value.config) {
                yaml += jst::fmt::format("    {}: '{}'\n", configKey,
                                         escapeSingleQuotedScalar(
                                             ParseLinkDomain(configValue, domain)));
            }
        }

        if (!value.input.empty()) {
            yaml += "  input:\n";
            for (const auto& [inputKey, inputValue] : value.input) {
                yaml += jst::fmt::format("    {}: {}\n", inputKey, ParseLinkDomain(inputValue, domain));
            }
        }
    }

    JST_TRACE("{}", yaml);

    std::vector<char> blob;
    std::copy(yaml.begin(), yaml.end(), std::back_inserter(blob));
    return blob;
}

Result Superluminal::box(const std::string& title, const Mosaic& mosaic, const std::function<void()>& callback) {
    JST_DEBUG("[SUPERLUMINAL] Registering new box called '{}'.", title);

    // Check boundaries.

    if (!impl->initialized) {
        JST_CHECK(initialize());
    }

    if (impl->running) {
        JST_FATAL("[SUPERLUMINAL] Can't register new box because the instance is already commited.");
        return Result::ERROR;
    }

    JST_CHECK(impl->validateMosaic(mosaic));
    JST_CHECK(impl->validateName(title));

    // Create plot state.

    auto& state = impl->plots[title];

    state.config.type = Type::Interface;
    state.callback = callback;
    state.mosaic = mosaic;
    state.name = title;

    JST_CHECK(impl->calculateMosaicParams(mosaic, state));

    JST_INFO("[SUPERLUMINAL] Created box '{}'.", state.name);
    return Result::SUCCESS;
}

Result Superluminal::text(const std::string& content) {
    ImGui::TextUnformatted(content.c_str());
    return Result::SUCCESS;
}

Result Superluminal::slider(const std::string& label, F32 min, F32 max, F32& value) {
    ImGui::SliderFloat(label.c_str(), &value, min, max);
    return Result::SUCCESS;
}


}  // namespace Jetstream
