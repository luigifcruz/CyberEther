#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstdint>

#include "jetstream/detail/instance_remote_supervisor.hh"
#include "jetstream/superluminal.hh"
#include "jetstream/macros.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/render/tools/imgui_markdown.hh"

#include "dmi_block.hh"

namespace Jetstream {

struct Superluminal::Impl {
    InstanceConfig config;
    std::shared_ptr<Instance> instance;
    std::shared_ptr<Flowgraph> flowgraph;
    bool initialized;
    bool running;

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
        std::shared_ptr<Jetstream::Block> block;
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
    static U64 BufferKey(const Tensor& buffer);
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
                    } else if (plot.block) {
                        for (const auto& surface : plot.block->surfaces()) {
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
    return impl->instance->polling();
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

    while (impl->instance->polling()) {
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

    // Create plot state.

    auto& state = impl->plots[name];

    state.config = config;
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
        auto& recipe = buffer_map[BufferKey(buf)];

        recipe.buffer = buf;
        recipe.source = state.config.source;
        recipe.display.insert(state.config.display);
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
                deviceNameStr = "cuda";
            }

            if ((recipe.buffer.device() == DeviceType::CPU) && (config.preferredDevice == DeviceType::CUDA)) {
                deviceNameStr = "cuda";
            }

            if (deviceNameStr.empty()) {
                JST_ERROR("[SUPERLUMINAL] Unsupported device conversion.");
                return Result::ERROR;
            }

            auto dtypeName = DataTypeToName(recipe.buffer.dtype());
            auto blob = GraphToYaml({
                {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), sourceDomain, hash),
                    {"duplicate", deviceNameStr, {std::string(dtypeName)}, {
                        {{"hostAccessible", "true"}}},
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
                auto winName = jst::fmt::format("win_{}", hash);
                auto invName = jst::fmt::format("inv_{}", hash);
                auto mulName = jst::fmt::format("win_mul_{}", hash);

                auto blob = GraphToYaml({
                    {winName,
                        {"window", GetDeviceName(config.preferredDevice), {"CF32"},
                            {{"size", jst::fmt::format("{}", recipe.buffer.size())}}, {}}},
                    {invName,
                        {"invert", GetDeviceName(config.preferredDevice), {"CF32"}, {},
                            {{"signal", jst::fmt::format("${{graph.{}.output.window}}", winName)}}}},
                    {mulName,
                        {"multiply", GetDeviceName(config.preferredDevice), {"CF32"}, {},
                            {{"a", jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), sourceDomain, hash)},
                            {"b", jst::fmt::format("${{graph.{}.output.signal}}", invName)}}}},
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
            case Type::Interface:
                break;
            case Type::Heat:
            case Type::Scatter:
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

    // Build graph.

    // TODO: Add Slice block in case of channel index.

    auto domain = (state.config.display == Domain::Time) ? "time" : "freq";
    auto hash = std::to_string(BufferKey(buf));
    auto outputPort = (state.config.display == state.config.source) ? "buffer" : "signal";
    auto port = jst::fmt::format("${{graph.data_{}_{}_{}.output.{}}}", GetDeviceName(config.preferredDevice), domain, hash, outputPort);

    bool isRealBuffer = (buf.dtype() == DataType::F32);

    // For real signals in time domain, we don't need amplitude conversion
    if (isRealBuffer && state.config.display == Domain::Time) {
        auto blob = GraphToYaml({
            {"scl",
                {"range", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"min", "-1"}, {"max", "1"}},
                    {{"signal", port}}}},
            {"lineplot",
                {"lineplot", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"averaging", averagingRate},
                     {"decimation", decimationRate}},
                    {{"signal", "${domain.scl.output.signal}"}}}},
        }, state.name);
        JST_CHECK(flowgraph->importFromBlob(blob));
    } else {
        std::string inputType = isRealBuffer ? "F32" : "CF32";

        auto blob = GraphToYaml({
            {"amp",
                {"amplitude", GetDeviceName(config.preferredDevice), {inputType, "F32"}, {},
                    {{"signal", port}}}},
            {"scl",
                {"range", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"min", "-100"}, {"max", "0"}},
                    {{"signal", "${domain.amp.output.signal}"}}}},
            {"lineplot",
                {"lineplot", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"averaging", averagingRate},
                     {"decimation", decimationRate}},
                    {{"signal", "${domain.scl.output.signal}"}}}},
        }, state.name);
        JST_CHECK(flowgraph->importFromBlob(blob));
    }

    // Update plot state.

    state.block = flowgraph->blockList().at(state.name + "_lineplot");

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

    // Build graph.

    auto graph = Graph{};
    auto hash = std::to_string(BufferKey(buf));
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
            {"slice", GetDeviceName(config.preferredDevice), {"CF32"},
                {{"slice", slice}},
                {{"buffer", port}}},
        });

        graph.push_back({
            "duplicate",
            {"duplicate", GetDeviceName(config.preferredDevice), {"CF32"}, {},
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
            {{"height", height}},
            {{"signal", "${domain.scl.output.signal}"}}},
    });

    JST_CHECK(flowgraph->importFromBlob(GraphToYaml(graph, state.name)));

    // Update plot state.

    state.block = flowgraph->blockList().at(state.name + "_waterfall");

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

U64 Superluminal::Impl::BufferKey(const Tensor& buffer) {
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
    hashCombine(seed, static_cast<U64>(buffer.rank()));
    for (U64 i = 0; i < buffer.rank(); ++i) {
        hashCombine(seed, buffer.shape(i));
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
                yaml += jst::fmt::format("    {}: '{}'\n", configKey, ParseLinkDomain(configValue, domain));
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

Result Superluminal::markdown(const std::string& content) {
    ImGui::MarkdownConfig mdConfig;
    ImGui::Markdown(content.c_str(), content.length(), mdConfig);
    return Result::SUCCESS;
}

}  // namespace Jetstream
