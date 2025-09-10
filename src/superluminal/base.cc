#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <fstream>

#include <stb_image.h>

#include "jetstream/superluminal.hh"
#include "jetstream/macros.hh"
#include "jetstream/store.hh"
#include "jetstream/blocks/manifest.hh"
#include "jetstream/memory/prototype.hh"

#include "dmi_module.hh"
#include "dmi_block.hh"

namespace Jetstream {

struct Superluminal::Impl {
    InstanceConfig config;
    Instance instance;

    bool initialized;
    bool running;

    std::atomic_flag computeSync{true};

    std::thread computeThread;
    std::thread presentThread;

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
    };

    std::unordered_map<std::string, PlotState> plots;

    // Image cache for loaded textures
    struct ImageData {
        std::shared_ptr<Render::Texture> texture;
        int width;
        int height;
    };
    std::unordered_map<std::string, ImageData> imageCache;
    std::unordered_set<std::string> failedImagePaths;

    Result createGraph();
    Result destroyGraph();

    Result validateBounds();
    Result validateMosaic(const Mosaic& mosaic);
    Result validateName(const std::string& name);

    Result calculateMosaicParams(const Mosaic& mosaic, PlotState& state);

    Result buildLinePlotGraph(PlotState& state);
    Result buildWaterfallPlotGraph(PlotState& state);

    Result loadImageFromFile(const std::string& filepath, ImageData& imageData);

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

    // Copy configuration to memory.

    impl->config = config;

    // Initialize the backend, viewport, and render.

    Backend::Config backendConfig {
        .deviceId = impl->config.deviceId,
        .remote = impl->config.remote,
    };

    Viewport::Config viewportConfig {
        .title = impl->config.windowTitle,
        .size = impl->config.interfaceSize,
        .codec = Viewport::VideoCodec::H264,
    };

    Render::Window::Config renderConfig {
        .scale =  impl->config.interfaceScale,
    };

    Instance::Config instanceConfig = {
        .preferredDevice = Device::None,
        .renderCompositor = false,
        .backendConfig = backendConfig,
        .viewportConfig = viewportConfig,
        .renderConfig = renderConfig,
    };

    JST_CHECK(impl->instance.build(instanceConfig));

    // Register custom blocks.

    Store::LoadBlocks([](Block::ConstructorManifest& constructorManifest,
                             Block::MetadataManifest& metadataManifest) {
        JST_TRACE("[SUPERLUMINAL] Registering custom blocks.");

        JST_BLOCKS_MANIFEST(
            Jetstream::Blocks::DynamicMemoryImport,
        )

        return Result::SUCCESS;
    });

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

    // Clear image cache and failed paths before destroying instance.

    impl->imageCache.clear();
    impl->failedImagePaths.clear();

    // Destroy instance.

    impl->instance.destroy();

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

    // Build graph and start instance.

    JST_CHECK(impl->instance.start());

    // Customize ImGui style.

    ImGui::GetStyle().Colors[ImGuiCol_WindowBg] = ImVec4(0.1, 0.1f, 0.1f, 1.0f);
    ImGui::GetStyle().WindowRounding = 0.0f;

    // Start the compute, present, and input threads.

    impl->computeThread = std::thread([&]{
        while (impl->instance.computing()) {
            impl->computeSync.wait(true);

            JST_CHECK_THROW(impl->instance.compute());

            impl->computeSync.test_and_set();
            impl->computeSync.notify_all();
        }

        JST_DEBUG("[SUPERLUMINAL] Compute thread safed.");
    });

    impl->presentThread = std::thread([&]{
        while (impl->instance.presenting()) {
            if (impl->instance.begin() == Result::SKIP) {
                continue;
            }

            if (impl->running) {
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
                    } else {
                        plot.block->drawView();
                    }

                    ImGui::End();
                }
            }

            JST_CHECK_THROW(impl->instance.present());
            if (impl->instance.end() == Result::SKIP) {
                continue;
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

    // Request to end the instance.

    impl->instance.reset();
    impl->instance.stop();

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
    return impl->instance.running();
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

    // Block until the instance is done inputting.

    while (impl->instance.running()) {
        impl->instance.viewport().waitEvents();
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

    if (wait) {
        impl->instance.viewport().waitEvents();
    } else {
        impl->instance.viewport().pollEvents();
    }

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

struct VariantBufferTypeVisitor {
    template<Device D, typename T>
    auto& operator()(const Tensor<D, T>& buffer) {
        return static_cast<const TensorPrototype&>(buffer);
    }
};

template<typename TargetType>
struct BufferTypeDetectorVisitor {
    bool matches = false;

    template<Device D, typename T>
    void operator()(const Tensor<D, T>&) {
        matches = std::is_same_v<T, TargetType>;
    }
};

Result Superluminal::Impl::createGraph() {
    JST_DEBUG("[SUPERLUMINAL] Create graph.");

    // Create empty flowgraph.

    JST_CHECK(instance.flowgraph().create());

    // Import memory buffers.

    struct InputMemoryRecipe {
        VariantBufferType buffer;
        Domain source;
        std::unordered_set<Domain> display;
    };

    std::unordered_map<U64, InputMemoryRecipe> buffer_map;

    for (auto& [_, state] : plots) {
        if (state.config.type == Type::Interface) {
            continue;
        }

        auto& prototype = std::visit(VariantBufferTypeVisitor{}, state.config.buffer);

        auto& recipe = buffer_map[prototype.hash()];

        recipe.buffer = state.config.buffer;
        recipe.source = state.config.source;
        recipe.display.insert(state.config.display);
    }

    for (auto& [hash, recipe] : buffer_map) {
        auto sourceDomain = (recipe.source == Domain::Time) ? "time" : "freq";
        auto conversionDomain = (recipe.source != Domain::Time) ? "time" : "freq";

        // Fulfills Time -> Time and Frequency -> Frequency conversions.

        auto convert = [&]<Device D, typename T>() {
            if (std::holds_alternative<Tensor<D, T>>(recipe.buffer)) {
                std::shared_ptr<Blocks::DynamicMemoryImport<D, void, T>> import;

                JST_CHECK(instance.addBlock(
                    import, jst::fmt::format("data_{}_{}_{}", GetDeviceName(D), sourceDomain, hash), {
                        .buffer = std::get<Tensor<D, T>>(recipe.buffer),
                    }, {}, {}
                ));

                if (D != config.preferredDevice) {
                    std::string deviceNameStr;

                    if ((D == Device::CUDA) and (config.preferredDevice == Device::CPU)) {
                        deviceNameStr = "cuda";
                    }

                    if ((D == Device::CPU) and (config.preferredDevice == Device::CUDA)) {
                        deviceNameStr = "cuda";
                    }

                    if (deviceNameStr.empty()) {
                        JST_ERROR("[SUPERLUMINAL] Unsupported device conversion.");
                        return Result::ERROR;
                    }

                    auto blob = GraphToYaml({
                        {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), sourceDomain, hash),
                            {"duplicate", deviceNameStr, {NumericTypeInfo<T>::name}, {
                                {{"hostAccessible", "true"}}},
                                {{"buffer", jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(D), sourceDomain, hash)}}}},
                    });

                    instance.flowgraph().importFromBlob(blob);
                }
            }
            return Result::SUCCESS;
        };

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        JST_CHECK(convert.operator()<Device::CPU, CF32>());
        JST_CHECK(convert.operator()<Device::CPU, F32>());
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        JST_CHECK(convert.operator()<Device::CUDA, CF32>());
        JST_CHECK(convert.operator()<Device::CUDA, F32>());
#endif

        // Check if a conversion is needed.

        if (recipe.display.contains(recipe.source) && recipe.display.size() == 1) {
            JST_DEBUG("[SUPERLUMINAL] Skipping conversion for '{}'.", hash);
            continue;
        }

        // Fulfills Time -> Frequency and Frequency -> Time conversions.

        auto& prototype = std::visit(VariantBufferTypeVisitor{}, recipe.buffer);
        auto forward = (recipe.source == Domain::Time && recipe.display.contains(Domain::Frequency));

        // Check if buffer is real (F32) or complex (CF32)
        BufferTypeDetectorVisitor<CF32> detector;
        std::visit(detector, recipe.buffer);
        bool isComplexBuffer = detector.matches;

        if (isComplexBuffer) {
            // Complex signal path - needs windowing, inversion, and multiplication
            if (config.preferredDevice == Device::CPU) {
                auto blob = GraphToYaml({
                    {"win",
                        {"window", GetDeviceName(config.preferredDevice), {"CF32"},
                            {{"size", jst::fmt::format("{}", prototype.size())}}, {}}},
                    {"inv",
                        {"invert", GetDeviceName(config.preferredDevice), {"CF32"}, {},
                            {{"buffer", "${graph.win.output.window}"}}}},
                    {"win_mul",
                        {"multiply", GetDeviceName(config.preferredDevice), {"CF32"}, {},
                            {{"factorA", jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), sourceDomain, hash)},
                            {"factorB", "${graph.inv.output.buffer}"}}}},
                    {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), conversionDomain, hash),
                        {"fft", GetDeviceName(config.preferredDevice), {"CF32", "CF32"},
                            {{"forward", jst::fmt::format("{}", (forward) ? "true" : "false")}},
                            {{"buffer", "${graph.win_mul.output.product}"}}}},
                });
                instance.flowgraph().importFromBlob(blob);
            } else {
                // TODO: The Multiply block doesn't support CUDA yet. This is a temporary bypass.
                auto blob = GraphToYaml({
                    {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), conversionDomain, hash),
                        {"fft", GetDeviceName(config.preferredDevice), {"CF32", "CF32"},
                            {{"forward", jst::fmt::format("{}", (forward) ? "true" : "false")}},
                            {{"buffer",  jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), sourceDomain, hash)}}}},
                });
                instance.flowgraph().importFromBlob(blob);
            }
        } else {
            // Real signal path - direct FFT without windowing for domain conversion
            auto blob = GraphToYaml({
                {jst::fmt::format("data_{}_{}_{}", GetDeviceName(config.preferredDevice), conversionDomain, hash),
                    {"fft", GetDeviceName(config.preferredDevice), {"F32", "F32"},
                        {{"forward", jst::fmt::format("{}", (forward) ? "true" : "false")}},
                        {{"buffer",  jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), sourceDomain, hash)}}}},
            });
            instance.flowgraph().importFromBlob(blob);
        }
    }

    // Create plots graph.

    for (auto& [name, state] : plots) {
        switch (state.config.type) {
            case Type::Line:
                buildLinePlotGraph(state);
                break;
            case Type::Waterfall:
                buildWaterfallPlotGraph(state);
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

    JST_CHECK(instance.flowgraph().destroy());

    return Result::SUCCESS;
}

Result Superluminal::Impl::buildLinePlotGraph(PlotState& state) {
    JST_DEBUG("[SUPERLUMINAL] Building line plot graph named '{}'.", state.name);

    // Access buffer metadata.

    auto& prototype = std::visit(VariantBufferTypeVisitor{}, state.config.buffer);

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
    auto hash = std::to_string(prototype.hash());
    auto port = jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), domain, hash);

    // Check if the source buffer is real (F32) or complex (CF32)
    BufferTypeDetectorVisitor<F32> realDetector;
    std::visit(realDetector, state.config.buffer);
    bool isRealBuffer = realDetector.matches;

    // For real signals in time domain, we don't need amplitude conversion
    if (isRealBuffer && state.config.display == Domain::Time) {
        auto blob = GraphToYaml({
            {"scl",
                {"scale", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"range", "[-1, 1]"}},  // Real signals typically have different scale range
                    {{"buffer", port}}}},
            {"lineplot",
                {"lineplot", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"averaging", averagingRate},
                     {"decimation", decimationRate}},
                    {{"buffer", "${domain.scl.output.buffer}"}}}},
        }, state.name);
        instance.flowgraph().importFromBlob(blob);
    } else {
        std::string inputType = isRealBuffer ? "F32" : "CF32";

        auto blob = GraphToYaml({
            {"amp",
                {"amplitude", GetDeviceName(config.preferredDevice), {inputType, "F32"}, {},
                    {{"buffer", port}}}},
            {"scl",
                {"scale", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"range", "[-100, 0]"}},
                    {{"buffer", "${domain.amp.output.buffer}"}}}},
            {"lineplot",
                {"lineplot", GetDeviceName(config.preferredDevice), {"F32"},
                    {{"averaging", averagingRate},
                     {"decimation", decimationRate}},
                    {{"buffer", "${domain.scl.output.buffer}"}}}},
        }, state.name);
        instance.flowgraph().importFromBlob(blob);
    }

    // Update plot state.

    state.block = instance.flowgraph().nodes()[{state.name + "_lineplot"}]->block;

    return Result::SUCCESS;
}

Result Superluminal::Impl::buildWaterfallPlotGraph(PlotState& state) {
    JST_DEBUG("[SUPERLUMINAL] Building waterfall plot graph named '{}'.", state.name);

    // Access buffer metadata.

    auto& prototype = std::visit(VariantBufferTypeVisitor{}, state.config.buffer);

    // Poll options.

    std::string height = "512";

    if (state.config.options.contains("height")) {
        auto h = std::get<I32>(state.config.options["height"]);
        JST_DEBUG("[SUPERLUMINAL] Height set to {}.", h);
        height = std::to_string(h);
    }

    // Build graph.

    auto graph = Graph{};
    auto hash = std::to_string(prototype.hash());
    auto domain = (state.config.display == Domain::Time) ? "time" : "freq";
    auto port = jst::fmt::format("${{graph.data_{}_{}_{}.output.buffer}}", GetDeviceName(config.preferredDevice), domain, hash);

    if (state.config.channelAxis != -1 && state.config.channelIndex != -1) {
        U64 axis = state.config.channelAxis;
        U64 index = state.config.channelIndex;

        // Parse slice string.

        std::string slice;
        for (U64 i = 0; i < prototype.rank(); i++) {
            if (i == axis) {
                slice += jst::fmt::format("{}", index);
            } else {
                slice += jst::fmt::format(":");
            }
            if (i != prototype.rank() - 1) {
                slice += ",";
            }
        }
        slice = jst::fmt::format("'[{}]'", slice);

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

    // Check if the source buffer is real (F32) or complex (CF32)
    BufferTypeDetectorVisitor<F32> realDetector;
    std::visit(realDetector, state.config.buffer);
    bool isRealBuffer = realDetector.matches;

    // For real signals in time domain, skip amplitude conversion
    if (isRealBuffer && state.config.display == Domain::Time) {
        graph.push_back({
            "scl",
            {"scale", GetDeviceName(config.preferredDevice), {"F32"},
                {{"range", "[-1, 1]"}},  // Real signals typically have different scale range
                {{"buffer", port}}},
        });
    } else {
        std::string inputType = isRealBuffer ? "F32" : "CF32";

        graph.push_back({
            "amp",
            {"amplitude", GetDeviceName(config.preferredDevice), {inputType, "F32"}, {},
                {{"buffer", port}}},
        });

        graph.push_back({
            "scl",
            {"scale", GetDeviceName(config.preferredDevice), {"F32"},
                {{"range", "[-100, 0]"}},
                {{"buffer", "${domain.amp.output.buffer}"}}},
        });
    }

    graph.push_back({
        "waterfall",
        {"waterfall", GetDeviceName(config.preferredDevice), {"F32"},
            {{"height", height}},
            {{"buffer", "${domain.scl.output.buffer}"}}},
    });

    instance.flowgraph().importFromBlob(GraphToYaml(graph, state.name));

    // Update plot state.

    state.block = instance.flowgraph().nodes()[{state.name + "_waterfall"}]->block;

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

// TODO: Upstream to `Flowgraph` class.
std::vector<char> Superluminal::Impl::GraphToYaml(const Graph& graph, std::string domain) {
    std::string yaml = jst::fmt::format(
        "---\n"
        "protocolVersion: 1.0.0\n"
        "cyberetherVersion: 1.0.0\n"
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
                yaml += jst::fmt::format("    {}: {}\n", configKey, ParseLinkDomain(configValue, domain));
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
    ImGui::Markdown(content.c_str(), content.length(), impl->instance.compositor().markdownConfig());
    return Result::SUCCESS;
}

Result Superluminal::Impl::loadImageFromFile(const std::string& filepath, ImageData& imageData) {
    if (failedImagePaths.contains(filepath)) {
        return Result::ERROR;
    }

    int width, height, channels;

    // Read file into memory
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        JST_ERROR("[SUPERLUMINAL] Failed to open image file '{}'", filepath);
        failedImagePaths.insert(filepath);
        return Result::ERROR;
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(fileSize);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        JST_ERROR("[SUPERLUMINAL] Failed to read image file '{}'", filepath);
        failedImagePaths.insert(filepath);
        return Result::ERROR;
    }

    unsigned char* data = stbi_load_from_memory(buffer.data(), static_cast<int>(fileSize), &width, &height, &channels, 4);
    if (!data) {
        const char* error = stbi_failure_reason();
        JST_ERROR("[SUPERLUMINAL] Failed to load image '{}': {}", filepath, error ? error : "Unknown error");
        failedImagePaths.insert(filepath);
        return Result::ERROR;
    }

    Render::Texture::Config config;
    config.size = {
        static_cast<U64>(width),
        static_cast<U64>(height)
    };
    config.buffer = static_cast<uint8_t*>(data);

    JST_CHECK(instance.window().build(imageData.texture, config));
    JST_CHECK(imageData.texture->create());

    imageData.width = width;
    imageData.height = height;

    stbi_image_free(data);

    JST_INFO("[SUPERLUMINAL] Successfully loaded image '{}' ({}x{})", filepath, width, height);
    return Result::SUCCESS;
}

Result Superluminal::image(const std::string& filepath, F32 width, F32 height, bool fitToWindow) {
    auto it = impl->imageCache.find(filepath);
    if (it == impl->imageCache.end()) {
        Impl::ImageData imageData;
        JST_CHECK(impl->loadImageFromFile(filepath, imageData));
        impl->imageCache[filepath] = imageData;
        it = impl->imageCache.find(filepath);
    }

    const auto& imageData = it->second;

    // Calculate display size
    ImVec2 displaySize;

    if (fitToWindow) {
        // Fit to available content region while preserving aspect ratio
        ImVec2 availableSize = ImGui::GetContentRegionAvail();
        F32 imageAspectRatio = static_cast<F32>(imageData.width) / static_cast<F32>(imageData.height);
        F32 windowAspectRatio = availableSize.x / availableSize.y;

        if (imageAspectRatio > windowAspectRatio) {
            displaySize = ImVec2(availableSize.x, availableSize.x / imageAspectRatio);
        } else {
            displaySize = ImVec2(availableSize.y * imageAspectRatio, availableSize.y);
        }
    } else if (width <= 0 && height <= 0) {
        // Use original size
        displaySize = ImVec2(static_cast<F32>(imageData.width), static_cast<F32>(imageData.height));
    } else if (width <= 0) {
        // Calculate width based on height and aspect ratio
        F32 aspectRatio = static_cast<F32>(imageData.width) / static_cast<F32>(imageData.height);
        displaySize = ImVec2(height * aspectRatio, height);
    } else if (height <= 0) {
        // Calculate height based on width and aspect ratio
        F32 aspectRatio = static_cast<F32>(imageData.height) / static_cast<F32>(imageData.width);
        displaySize = ImVec2(width, width * aspectRatio);
    } else {
        // Use specified dimensions
        displaySize = ImVec2(width, height);
    }

    ImGui::Image(ImTextureRef(imageData.texture->raw()), displaySize);
    return Result::SUCCESS;
}

}  // namespace Jetstream
