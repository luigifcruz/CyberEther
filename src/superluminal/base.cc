#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

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
        PlotConfig config;
        std::shared_ptr<Jetstream::Block> block;
    };

    std::unordered_map<std::string, PlotState> plots;

    Result createGraph();
    Result destroyGraph();

    Result buildLinePlotGraph(PlotState& state);

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
        .headless = impl->config.headless,
    };

    Viewport::Config viewportConfig {
        .title = impl->config.windowTitle,
        .size = impl->config.interfaceSize,
        .endpoint = impl->config.endpoint,
        .codec = Viewport::VideoCodec::H264,
    };

    Render::Window::Config renderConfig {
        .scale =  impl->config.interfaceScale,
    };

    Instance::Config instanceConfig = {
        .preferredDevice = Device::None,
        .enableCompositor = false,
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

    // Build graph and start instance.

    JST_CHECK(impl->instance.start());

    // Customize ImGui style.

    ImGui::GetStyle().Colors[ImGuiCol_WindowBg] = ImVec4(0.1, 0.1f, 0.1f, 1.0f);

    // Create graph.

    JST_CHECK(impl->createGraph());

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
                        viewport->WorkSize.x / impl->mosaicDims.x,
                        viewport->WorkSize.y / impl->mosaicDims.y
                    };

                    ImVec2 workPos = {
                        viewport->WorkPos.x + (workSize.x * plot.mosaicOffset.x),
                        viewport->WorkPos.y + (workSize.y * plot.mosaicOffset.y)
                    };

                    ImGui::SetNextWindowPos(workPos);
                    ImGui::SetNextWindowSize(workSize);

                    ImGui::Begin(plot.name.c_str(), nullptr, flags);
                    plot.block->drawView();
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
    impl->computeSync.clear();
    impl->computeSync.notify_all();

    return Result::SUCCESS;
}

bool Superluminal::presenting() {
    return impl->instance.viewport().keepRunning();
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

    while (impl->instance.viewport().keepRunning()) {
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

    // Validate mosaic size.

    Extent2D<U8> mosaicDims;
    mosaicDims.y = mosaic.size();

    if (mosaicDims.y == 0) {
        JST_FATAL("[SUPERLUMINAL] Mosaic should be a 2D matrix. Currently: '{}'. Example: '{{0}}', or '{{{{0, 0}}, {{0, 1}}}}'.", mosaic);
        return Result::ERROR;
    }

    mosaicDims.x = mosaic[0].size();
    for (auto& column : mosaic) {
        if (mosaicDims.x != column.size()) {
            JST_FATAL("[SUPERLUMINAL] All mosaic rows should have the same size");
            return Result::ERROR;
        }
    }

    if (impl->plots.size() == 0) {
        impl->mosaicDims = mosaicDims;
    } else {
        if (impl->mosaicDims != mosaicDims) {
            JST_FATAL("[SUPERLUMINAL] The mosaic dimensions of all plots need to be the same.");
            return Result::ERROR;
        }
    }

    // Check plot name and create state.

    if (impl->plots.contains(name)) {
        JST_FATAL("[SUPERLUMINAL] Plot with name '{}' already exists.", name);
        return Result::ERROR;
    }

    auto& state = impl->plots[name];

    state.config = config;
    state.mosaic = mosaic;
    state.name = name;

    // Calculate mosaic offset.

    for (U8 x = 0; x < impl->mosaicDims.x; x++) {
        for (U8 y = 0; y < impl->mosaicDims.y; y++) {
            if (mosaic[y][x] != 0) {
                state.mosaicOffset.x = x;
                state.mosaicOffset.y = y;
            }
        }
    }

    JST_DEBUG("[SUPERLUMINAL] Mosaic offsets for plot '{}' is (X: {}, Y: {}).", state.name,
                                                                                state.mosaicOffset.x,
                                                                                state.mosaicOffset.y);

    JST_INFO("[SUPERLUMINAL] Created plot '{}'.", state.name);
    return Result::SUCCESS;
}

struct VariantBufferTypeVisitor {
    template<Device D, typename T>
    auto& operator()(const Tensor<D, T>& buffer) {
        return static_cast<const TensorPrototype&>(buffer);
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

        {
            std::shared_ptr<Blocks::DynamicMemoryImport<Device::CPU, void, CF32>> import;

            JST_CHECK(instance.addBlock(
                import, jst::fmt::format("data_{}_{}", sourceDomain, hash), {
                    .buffer = std::get<Tensor<Device::CPU, CF32>>(recipe.buffer),
                }, {}, {}
            ));
        }

        // Check if a conversion is needed.

        if (recipe.display.contains(recipe.source) && recipe.display.size() == 1) {
            JST_DEBUG("[SUPERLUMINAL] Skipping conversion for '{}'.", hash);
            continue;
        }

        // Fulfills Time -> Frequency and Frequency -> Time conversions.

        auto& prototype = std::visit(VariantBufferTypeVisitor{}, recipe.buffer);
        auto forward = (recipe.source == Domain::Time && recipe.display.contains(Domain::Frequency));

        auto blob = GraphToYaml({
            {"win",
                {"window", "cpu", {"CF32"},
                    {{"size", jst::fmt::format("{}", prototype.size())}}, {}}},
            {"inv",
                {"invert", "cpu", {"CF32"}, {},
                    {{"buffer", "${graph.win.output.window}"}}}},
            {"win_mul",
                {"multiply", "cpu", {"CF32"}, {},
                    {{"factorA", jst::fmt::format("${{graph.data_{}_{}.output.buffer}}", sourceDomain, hash)},
                     {"factorB", "${graph.inv.output.buffer}"}}}},
            {jst::fmt::format("data_{}_{}", conversionDomain, hash),
                {"fft", "cpu", {"CF32", "CF32"},
                    {{"forward", jst::fmt::format("{}", (forward) ? "true" : "false")}},
                    {{"buffer", "${graph.win_mul.output.product}"}}}},
        });

        instance.flowgraph().importFromBlob(blob);
    }

    // Create plots graph.

    for (auto& [name, state] : plots) {
        switch (state.config.type) {
            case Type::Line:
                buildLinePlotGraph(state);
                break;
            case Type::Heat:
            case Type::Waterfall:
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

    // Build graph.

    // TODO: Add Slice block in case of channel index.

    auto domain = (state.config.display == Domain::Time) ? "time" : "freq";
    auto hash = std::to_string(prototype.hash());

    auto blob = GraphToYaml({
        {"amp",
            {"amplitude", "cpu", {"CF32", "F32"}, {},
                {{"buffer", jst::fmt::format("${{graph.data_{}_{}.output.buffer}}", domain, hash)}}}},
        {"scl",
            {"scale", "cpu", {"F32"},
                {{"range", "[-100, 0]"}},
                {{"buffer", "${domain.amp.output.buffer}"}}}},
        {"lineplot",
            {"lineplot", "cpu", {"F32"}, {},
                {{"buffer", "${domain.scl.output.buffer}"}}}},
    }, state.name);

    instance.flowgraph().importFromBlob(blob);

    // Update plot state.

    state.block = instance.flowgraph().nodes()[{state.name + "_lineplot"}]->block;

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

}  // namespace Jetstream