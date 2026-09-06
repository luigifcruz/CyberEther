#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "jetstream/domains/visualization/signal_view/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/testing.hh"
#include "flowgraph_fixture.hh"

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct SignalViewImplAccess : Modules::SignalViewImpl {
    static auto signalPointsMember() {
        return &SignalViewImplAccess::signalPoints;
    }

    static auto maxHoldPointsMember() {
        return &SignalViewImplAccess::maxHoldPoints;
    }

    static auto maxHoldWarmupBlocksMember() {
        return &SignalViewImplAccess::maxHoldWarmupBlocks;
    }

    static auto waterfallBinsMember() {
        return &SignalViewImplAccess::waterfallBins;
    }

    static auto waterfallHistoryMember() {
        return &SignalViewImplAccess::waterfallHistory;
    }

    static auto axisMember() {
        return &SignalViewImplAccess::axis;
    }

    static auto renderSurfaceMember() {
        return &SignalViewImplAccess::renderSurface;
    }

    static auto waterfallUniformBufferMember() {
        return &SignalViewImplAccess::waterfallUniformBuffer;
    }

    static auto interactionMember() {
        return &SignalViewImplAccess::interaction;
    }

    static auto splitterMember() {
        return &SignalViewImplAccess::splitter;
    }

    static auto processMouseEventsMember() {
        return &SignalViewImplAccess::processMouseEvents;
    }

    static auto configChangePendingMember() {
        return &SignalViewImplAccess::configChangePending;
    }
};

struct InteractiveSignalViewConfig : Block::Config {
    F32 splitRatio = 0.5f;
    JST_BLOCK_TYPE(interactive_signal_view_test);
    JST_BLOCK_DOMAIN("Test");
    JST_BLOCK_PARAMS(splitRatio);
    JST_BLOCK_DESCRIPTION("Interactive Signal View Test", "Test surface edits.", "Test surface edits.");
};

std::shared_ptr<Module> interactiveSignalView;

struct InteractiveSignalViewBlock : Block::Impl, DynamicConfig<InteractiveSignalViewConfig> {
    std::shared_ptr<Modules::SignalView> config = std::make_shared<Modules::SignalView>();

    Result define() override { return defineInterfaceInput("signal", "Signal", "Signal"); }
    Result configure() override {
        config->mode = "lineplot_waterfall";
        config->waterfallHeight = 8;
        config->splitRatio = splitRatio;
        return Result::SUCCESS;
    }
    Result create() override {
        JST_CHECK(moduleCreate("plot", config, inputs()));
        JST_CHECK(moduleBindConfigEdit("plot", "splitRatio", "splitRatio"));
        interactiveSignalView = moduleHandle("plot");
        return Result::SUCCESS;
    }
};

JST_REGISTER_BLOCK(InteractiveSignalViewBlock, {"signal_view"});

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
// Build the axis's CPU-side label state without creating GPU resources or a
// viewport. The Vulkan resource constructors only describe pending resources.
class LabelTestWindow final : public Render::Window {
 public:
    LabelTestWindow() : Window(Config{}) {}

    const Stats& stats() const override { return windowStats; }
    std::string info() const override { return "LabelTestWindow"; }
    constexpr DeviceType device() const override { return DeviceType::Vulkan; }

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }
    Result unbindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }
    Result underlyingCreate() override { return Result::SUCCESS; }
    Result underlyingDestroy() override { return Result::SUCCESS; }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override { return Result::SUCCESS; }
    Result underlyingSynchronize() override { return Result::SUCCESS; }

 private:
    Stats windowStats{};
};

class LabelTestAxis final : public Render::Components::Axis {
 public:
    explicit LabelTestAxis(const Config& config) : Axis(config) {}

    // Label updates are real; only GPU uploads are omitted.
    Result present() override { return Result::SUCCESS; }
};

class LabelTestSurface final : public Render::Surface {
 public:
    LabelTestSurface() : Surface(Config{}) {}

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }

    const Extent2D<U64>& size(const Extent2D<U64>& nextSize) override {
        viewSize = nextSize;
        return viewSize;
    }

 private:
    Extent2D<U64> viewSize;
};

class LabelTestBuffer final : public Render::Buffer {
 public:
    LabelTestBuffer() : Buffer(Config{}) {}

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
};
#endif

std::vector<F32> ReadTensor(const Tensor& tensor,
                            const char* name) {
    Tensor hostTensor;
    const Tensor* readable = &tensor;
    if (tensor.device() != DeviceType::CPU) {
        if (hostTensor.create(DeviceType::CPU, tensor) != Result::SUCCESS) {
            throw std::runtime_error(std::string(name) + " is not host accessible");
        }
        readable = &hostTensor;
    }

    const F32* data = readable->data<F32>();
    return {data, data + readable->size()};
}

std::vector<F32> ReadSignalPoints(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }

    return ReadTensor(impl->*SignalViewImplAccess::signalPointsMember(),
                      "signal points");
}

void RequireSignalViewValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::SignalView& config,
                                      const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view",
                                  impl.device,
                                  impl.runtime,
                                  impl.provider,
                                  module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

void RequireSignalViewValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::SignalView& config,
                                      const DataType dtype,
                                      const Shape& shape) {
    Tensor input;
    if (shape.empty()) {
        REQUIRE(input.create(impl.device, dtype, {1}) == Result::SUCCESS);
        REQUIRE(input.squeezeDims(0) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }
    RequireSignalViewValidationError(impl, config, input);
}

std::vector<F32> ReadMaxHoldPoints(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }

    return ReadTensor(impl->*SignalViewImplAccess::maxHoldPointsMember(),
                      "max hold points");
}

std::vector<F32> ReadWaterfallBins(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }

    return ReadTensor(impl->*SignalViewImplAccess::waterfallBinsMember(),
                      "waterfall history");
}

const Modules::WaterfallHistory&
ReadWaterfallHistory(const std::shared_ptr<Module>& module) {
    const auto* impl = module->getImpl<Modules::SignalViewImpl>();
    if (!impl) {
        throw std::runtime_error(
            "signal view implementation is unavailable");
    }
    return impl->*SignalViewImplAccess::waterfallHistoryMember();
}

struct SignalViewSnapshot {
    std::vector<F32> signalPoints;
    std::vector<F32> waterfallBins;
};

SignalViewSnapshot ComputeSignalViewSnapshot(const Registry::ModuleRegistration& impl,
                                             const Tensor& cpuInput) {
    Tensor input;
    if (impl.device == DeviceType::CPU) {
        input = cpuInput;
    } else {
        REQUIRE(input.create(impl.device, cpuInput) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["signal"].requested("source", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view",
                                  impl.device,
                                  impl.runtime,
                                  impl.provider,
                                  module) == Result::SUCCESS);

    Modules::SignalView config;
    config.mode = "lineplot_waterfall";
    config.fill = false;
    config.waterfallHeight = 2;
    REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

    Runtime runtime("signal_view", impl.device, impl.runtime);
    REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
    std::unordered_set<std::string> skipped;
    std::unordered_set<std::string> failed;
    REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

    SignalViewSnapshot snapshot{
        .signalPoints = ReadSignalPoints(module),
        .waterfallBins = ReadWaterfallBins(module),
    };
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
    return snapshot;
}

void ApplyReferenceRows(std::vector<F32>& ring,
                        U64& writeIndex,
                        const Tensor& input,
                        const U64 height) {
    const U64 batches = input.shape(0);
    const U64 width = input.shape(1);
    const U64 retained = std::min(batches, height);
    const U64 sourceRow = batches - retained;
    const U64 destinationRow = (writeIndex + (sourceRow % height)) % height;
    const F32* inputData = input.data<F32>();
    for (U64 row = 0; row < retained; ++row) {
        std::copy_n(inputData + (sourceRow + row) * width,
                    width,
                    ring.data() + ((destinationRow + row) % height) * width);
    }
    writeIndex = (writeIndex + (batches % height)) % height;
}

}  // namespace

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
TEST_CASE("Axis vertical scale can move the divider without recreating resources",
          "[modules][signal_view][split][axis]") {
    LabelTestWindow window;
    Render::Components::Axis::Config config;
    config.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    LabelTestAxis axis(config);
    REQUIRE(axis.create(&window) == Result::SUCCESS);
    REQUIRE(axis.updatePixelSize({2.0f / 1000.0f, 2.0f / 800.0f}) == Result::SUCCESS);
    for (const F32 ratio : {0.1f, 0.35f, 0.9f, 1.0f, 0.5f}) {
        REQUIRE(axis.updateVerticalScale(ratio) == Result::SUCCESS);
        REQUIRE(axis.getConfig().verticalScale == ratio);
        REQUIRE(axis.currentHorizontalLineCount() >= 3);
    }
    REQUIRE(axis.updateVerticalScale(0.0f) == Result::ERROR);
    REQUIRE(axis.updateVerticalScale(std::numeric_limits<F32>::quiet_NaN()) == Result::ERROR);
    REQUIRE(axis.getConfig().verticalScale == 0.5f);
    REQUIRE(axis.destroy(&window) == Result::SUCCESS);
}

TEST_CASE("Axis created in split mode reserves full height tick geometry",
          "[modules][signal_view][split][axis][capacity]") {
    LabelTestWindow window;
    Render::Components::Axis::Config config;
    config.verticalScale = 0.5f;
    config.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    LabelTestAxis axis(config);
    REQUIRE(axis.create(&window) == Result::SUCCESS);
    REQUIRE(axis.updatePixelSize({2.0f / 16384.0f, 2.0f / 16384.0f}) == Result::SUCCESS);
    REQUIRE(axis.currentVerticalLineCount() == 65);
    REQUIRE(axis.currentHorizontalLineCount() == 17);
    REQUIRE(axis.setShowFrameTicks(true) == Result::SUCCESS);

    Render::Surface::Config resources;
    REQUIRE(axis.surfaceUnderlay(resources) == Result::SUCCESS);
    REQUIRE(resources.kernels.size() == 1);
    const auto& kernel = resources.kernels.front()->getConfig();
    REQUIRE(kernel.buffers.size() == 3);
    const auto& points = kernel.buffers[1].first->getConfig();
    const auto& vertices = kernel.buffers[2].first->getConfig();

    // Full-height mode has ticks on both horizontal edges. Reserve those
    // ticks, the interior grid, the frame, and space for a divider.
    constexpr U64 fullHeightLines = (65 - 2) + (17 - 2) +
                                    2 * (65 - 2) + 2 * (17 - 2) +
                                    8 * (65 - 1) + 8 * (17 - 1) + 4;
    REQUIRE(points.size >= (fullHeightLines + 1) * 4);
    REQUIRE(vertices.size >= (fullHeightLines + 1) * 24);
    REQUIRE(std::get<0>(kernel.gridSize) >= fullHeightLines + 1);
    const auto* originalPoints = points.buffer;

    for (const F32 ratio : {1.0f, 0.25f, 1.0f}) {
        REQUIRE(axis.updateVerticalScale(ratio) == Result::SUCCESS);
        REQUIRE(axis.currentVerticalLineCount() == 65);
        REQUIRE(axis.currentHorizontalLineCount() == 17);
        REQUIRE(points.buffer == originalPoints);
        if (ratio == 1.0f) {
            const auto* data = static_cast<const F32*>(points.buffer);
            const std::array<F32, 16> frame = {
                -1.0f, -1.0f,  1.0f, -1.0f,
                -1.0f,  1.0f,  1.0f,  1.0f,
                -1.0f, -1.0f, -1.0f,  1.0f,
                 1.0f, -1.0f,  1.0f,  1.0f,
            };
            REQUIRE(std::equal(frame.begin(), frame.end(), data + (fullHeightLines - 4) * 4));
        }
    }
    REQUIRE(axis.destroy(&window) == Result::SUCCESS);
}

TEST_CASE("Standalone waterfall presents live metadata without view changes",
          "[modules][signal_view][waterfall][present][metadata][regression]") {
    Tensor input(DeviceType::CPU, DataType::F32, {32});
    F32 frequency = 100.0e6f;
    F32 sampleRate = 2.0e6f;
    F32 observedFrequency = 0.0f;
    F32 observedSampleRate = 0.0f;
    U64 frequencyReads = 0;
    U64 sampleRateReads = 0;
    REQUIRE(input.setDerivedAttribute("frequency", [&]() -> std::any {
        ++frequencyReads;
        observedFrequency = frequency;
        return frequency;
    }) == Result::SUCCESS);
    REQUIRE(input.setDerivedAttribute("sampleRate", [&]() -> std::any {
        ++sampleRateReads;
        observedSampleRate = sampleRate;
        return sampleRate;
    }) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("source", "signal");
    inputs["signal"].tensor = input;
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view", DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic", module) ==
            Result::SUCCESS);
    Modules::SignalView config;
    config.mode = "waterfall";
    config.waterfallHeight = 4;
    REQUIRE(module->create("waterfall", config, inputs) == Result::SUCCESS);

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.showInteriorGrid = false;
    axisConfig.font = std::make_shared<Render::Components::Font>(
        Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);

    auto* impl = module->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    impl->*SignalViewImplAccess::axisMember() = axis;
    impl->*SignalViewImplAccess::renderSurfaceMember() =
        std::make_shared<LabelTestSurface>();
    impl->*SignalViewImplAccess::waterfallUniformBufferMember() =
        std::make_shared<LabelTestBuffer>();
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);

    // Ignore validation/creation reads. Present must reread live metadata even
    // without new samples, mouse events, or surface resize/placement events.
    const auto present = [&] {
        frequencyReads = 0;
        sampleRateReads = 0;
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        CHECK_FALSE((impl->*SignalViewImplAccess::interactionMember()).viewChanged);
        CHECK(frequencyReads > 0);
        CHECK(sampleRateReads > 0);
        CHECK(observedFrequency == frequency);
        CHECK(observedSampleRate == sampleRate);
    };
    present();
    frequency = 101.5e6f;
    present();
    sampleRate = 4.0e6f;
    present();

    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}
#endif

TEST_CASE_METHOD(FlowgraphFixture, "Signal View drag requests round trip through the owning block",
                 "[modules][signal_view][split][config-edits]") {
    TestFlowgraph::SyntheticSourceBlockConfig source;
    source.bufferSize = 32;
    REQUIRE(flowgraph->blockCreate("source", source, {}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].requested("source", "signal");
    REQUIRE(flowgraph->blockCreate("plot", InteractiveSignalViewConfig{}, inputs) == Result::SUCCESS);
    const auto original = interactiveSignalView;
    auto* impl = original->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    auto& splitter = impl->*SignalViewImplAccess::splitterMember();
    auto& interaction = impl->*SignalViewImplAccess::interactionMember();
    const auto process = [&] {
        (impl->*SignalViewImplAccess::processMouseEventsMember())({0.8f, 0.75f});
    };
    const auto pending = [&] {
        return (impl->*SignalViewImplAccess::configChangePendingMember())();
    };
    MouseEvent event{};
    event.type = MouseEventType::Click;
    event.button = MouseButton::Left;
    event.position = {0.5f, 0.5f};
    original->surface()->pushMouseEvent(event);
    process();
    REQUIRE(splitter.dragging);
    REQUIRE_FALSE(interaction.dragging);

    event.type = MouseEventType::Move;
    event.position = {0.7f, 0.65f};
    original->surface()->pushMouseEvent(event);
    process();
    REQUIRE(splitter.ratio == Catch::Approx(0.7f));
    REQUIRE_FALSE(pending());
    REQUIRE(interaction.offset == 0.0f);
    REQUIRE(std::any_cast<F32>(viewBlock("plot").config.at("splitRatio")) == 0.5f);

    SECTION("release inside the surface") {
        event.position.x = 0.7f;
    }
    SECTION("release outside the surface") {
        event.position.x = 1.2f;
    }
    event.type = MouseEventType::Release;
    original->surface()->pushMouseEvent(event);
    process();
    REQUIRE_FALSE(splitter.dragging);
    REQUIRE(pending());
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    process();
    REQUIRE_FALSE(pending());
    REQUIRE(interactiveSignalView == original);
    REQUIRE(std::any_cast<F32>(viewBlock("plot").config.at("splitRatio")) == Catch::Approx(0.7f));
    REQUIRE(impl->splitRatio == splitter.ratio);

    const auto committedRatio = splitter.ratio;
    event.type = MouseEventType::Move;
    event.position = {0.2f, 0.2f};
    original->surface()->pushMouseEvent(event);
    process();
    REQUIRE_FALSE(splitter.dragging);
    REQUIRE_FALSE(pending());
    REQUIRE(splitter.ratio == committedRatio);

    Parser::Map sidebar;
    sidebar["splitRatio"] = F32{0.4f};
    REQUIRE(flowgraph->blockReconfigure("plot", sidebar) == Result::SUCCESS);
    process();
    REQUIRE(splitter.ratio == 0.4f);
    REQUIRE_FALSE(pending());
}

TEST_CASE("Signal View splitter hit testing uses the padded plot and commits on release",
          "[modules][signal_view][split]") {
    using namespace Modules::detail;
    const Extent2D<U64> size{1000, 800};
    const auto layout = CalculateSignalViewPanels({0.8f, 0.75f}, size, 0.25f);
    SignalViewSplitInteraction split;
    split.ratio = 0.25f;
    bool commit = false;
    MouseEvent event{};
    event.type = MouseEventType::Click;
    event.button = MouseButton::Left;
    event.position = {0.5f, (layout.waterfall.y + 4.0f) / size.y};

    REQUIRE_FALSE(split.process(event, layout, size, 1.0f, false, commit));
    REQUIRE(split.process(event, layout, size, 1.0f, true, commit));
    REQUIRE(split.dragging);
    REQUIRE_FALSE(commit);
    REQUIRE(split.ratio == 0.25f);

    event.type = MouseEventType::Move;
    event.position.y = (layout.plot.y + layout.plot.height * 0.4f + 4.0f) / size.y;
    REQUIRE(split.process(event, layout, size, 1.0f, true, commit));
    REQUIRE(split.ratio == Catch::Approx(0.4f));
    REQUIRE_FALSE(commit);

    // Use the release position even if there was no preceding Move event.
    event.type = MouseEventType::Release;
    event.position.y = (layout.plot.y + layout.plot.height * 0.6f + 4.0f) / size.y;
    REQUIRE(split.process(event, layout, size, 1.0f, true, commit));
    REQUIRE(split.ratio == Catch::Approx(0.6f));
    REQUIRE_FALSE(split.dragging);
    REQUIRE(commit);
}

TEST_CASE("Signal View splitter captures only the divider and bounds out of view drags",
          "[modules][signal_view][split]") {
    using namespace Modules::detail;
    const Extent2D<U64> size{1000, 800};
    const auto layout = CalculateSignalViewPanels({0.8f, 0.75f}, size, 0.5f);
    SignalViewSplitInteraction split;
    bool commit = false;
    MouseEvent event{};
    event.type = MouseEventType::Click;
    event.button = MouseButton::Left;
    event.position = {0.01f, 0.5f};
    REQUIRE_FALSE(split.process(event, layout, size, 1.0f, true, commit));
    event.position = {0.5f, 0.2f};
    REQUIRE_FALSE(split.process(event, layout, size, 1.0f, true, commit));
    event.position.y = (layout.waterfall.y + 10.0f) / size.y;
    REQUIRE_FALSE(split.process(event, layout, size, 1.0f, true, commit));
    REQUIRE(split.process(event, layout, size, 2.0f, true, commit));

    event.type = MouseEventType::Move;
    event.position = {2.0f, -1.0f};
    REQUIRE(split.process(event, layout, size, 2.0f, true, commit));
    REQUIRE(split.ratio == MinSplitRatio);
    event.position.y = 2.0f;
    REQUIRE(split.process(event, layout, size, 2.0f, true, commit));
    REQUIRE(split.ratio == MaxSplitRatio);
    event.position.y = std::numeric_limits<F32>::quiet_NaN();
    REQUIRE(split.process(event, layout, size, 2.0f, true, commit));
    REQUIRE(split.ratio == MaxSplitRatio);
    event.type = MouseEventType::Leave;
    REQUIRE(split.process(event, layout, size, 2.0f, true, commit));
    REQUIRE_FALSE(commit);
    REQUIRE_FALSE(split.dragging);
}

TEST_CASE("Signal View panel rectangles tile the plot without gaps or overlaps",
          "[modules][signal_view][split]") {
    for (const auto height : {U64{0}, U64{1}, U64{2}, U64{255}, U64{512}}) {
        for (const auto ratio : {0.1f, 0.35f, 0.5f, 0.9f}) {
            const auto panels = Modules::detail::CalculateSignalViewPanels(
                {0.9f, 0.8f}, {512, height}, ratio);
            REQUIRE(panels.line.y == panels.plot.y);
            REQUIRE(panels.line.height + panels.waterfall.height == panels.plot.height);
            REQUIRE(panels.waterfall.y == panels.line.y + panels.line.height);
            REQUIRE(panels.waterfall.y + panels.waterfall.height ==
                    panels.plot.y + panels.plot.height);
            REQUIRE(panels.lineFraction > 0.0f);
            REQUIRE(panels.lineFraction < 1.0f);
        }
    }
    const auto tiny = Modules::detail::CalculateSignalViewPanels({-1.0f, -1.0f}, {8, 8}, 0.5f);
    REQUIRE(tiny.plot.width == 0);
    REQUIRE(tiny.plot.height == 0);
}

TEST_CASE("Signal View split reconfiguration preserves trace and waterfall history",
          "[modules][signal_view][split][reconfigure]") {
    for (const auto& implementation : Registry::ListAvailableModules("signal_view")) {
        DYNAMIC_SECTION("Device: " << implementation.device) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {8}) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.75f);
            TensorMap inputs;
            inputs["signal"].tensor = input;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view", implementation.device,
                                          implementation.runtime, implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.waterfallHeight = 8;
            config.averaging = 2;
            config.maxHold = true;
            REQUIRE(module->create("plot", config, inputs) == Result::SUCCESS);
            Runtime runtime("plot", implementation.device, implementation.runtime);
            REQUIRE(runtime.create({{"plot", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped, failed;
            for (U64 cycle = 0; cycle < 3; ++cycle) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }
            const auto trace = ReadSignalPoints(module);
            const auto hold = ReadMaxHoldPoints(module);
            const auto bins = ReadWaterfallBins(module);
            const auto history = ReadWaterfallHistory(module);
            auto* impl = module->getImpl<Modules::SignalViewImpl>();
            const auto warmup = impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember();
            const auto signalId = (impl->*SignalViewImplAccess::signalPointsMember()).id();

            Parser::Map edit;
            edit["splitRatio"] = F32{0.35f};
            REQUIRE(module->reconfigure(edit, true) == Result::SUCCESS);
            REQUIRE(static_cast<const Modules::SignalView&>(module->config()).splitRatio == 0.5f);
            REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
            REQUIRE(static_cast<const Modules::SignalView&>(module->config()).splitRatio == 0.35f);
            REQUIRE(ReadSignalPoints(module) == trace);
            REQUIRE(ReadMaxHoldPoints(module) == hold);
            REQUIRE(ReadWaterfallBins(module) == bins);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == history.writeIndex);
            REQUIRE(ReadWaterfallHistory(module).dirtyRows == history.dirtyRows);
            REQUIRE((impl->*SignalViewImplAccess::signalPointsMember()).id() == signalId);
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == warmup);

            for (const F32 invalid : {0.0f, 1.0f, std::numeric_limits<F32>::infinity(),
                                       std::numeric_limits<F32>::quiet_NaN()}) {
                edit["splitRatio"] = invalid;
                REQUIRE(module->reconfigure(edit) == Result::ERROR);
                REQUIRE(static_cast<const Modules::SignalView&>(module->config()).splitRatio == 0.35f);
            }
            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View module supports every visualization mode",
          "[modules][signal_view]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    const std::array<const char*, 3> modes = {
        "lineplot",
        "waterfall",
        "lineplot_waterfall",
    };

    for (const auto& implementation : implementations) {
        for (const char* mode : modes) {
            DYNAMIC_SECTION("Device: " << implementation.device
                            << " Runtime: " << implementation.runtime
                            << " Mode: " << mode) {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = mode;
                config.averaging = 4;
                config.decimation = 2;
                config.waterfallHeight = 32;
                config.xLabel = "Frequency";
                config.amplitudeLabel = "Power";
                config.waterfallLabel = "History";
                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 128}) ==
                        Result::SUCCESS);
                REQUIRE(SetSignalAxes(input, {
                    .sample = Index{1},
                    .batch = Index{0},
                }) == Result::SUCCESS);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::SUCCESS);

                Tensor channels;
                REQUIRE(channels.create(DeviceType::CPU, DataType::F32, {128}) ==
                        Result::SUCCESS);
                REQUIRE(SetSignalAxes(channels, {
                    .channel = Index{0},
                }) == Result::SUCCESS);
                ctx.setInput("signal", channels);
                REQUIRE(ctx.run() == Result::SUCCESS);
            }
        }
    }
}

TEST_CASE("Signal View validation is mode aware",
          "[modules][signal_view][validation]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) ==
                    Result::SUCCESS);

            SECTION("invalid mode") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "invalid";
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("line settings") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "lineplot";
                config.decimation = 0;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);

                config.decimation = 1;
                config.averaging = 0;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("waterfall settings") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "waterfall";
                config.waterfallHeight = 0;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);

                config.waterfallHeight = 8193;
                ctx.setConfig(config);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("waterfall accepts one column") {
                TestContext ctx("signal_view", implementation.device,
                                implementation.runtime, implementation.provider);
                Modules::SignalView config;
                config.mode = "waterfall";
                ctx.setConfig(config);
                Tensor oneColumn;
                REQUIRE(oneColumn.create(DeviceType::CPU, DataType::F32, {1}) ==
                        Result::SUCCESS);
                ctx.setInput("signal", oneColumn);
                REQUIRE(ctx.run() == Result::SUCCESS);
            }
        }
    }
}

TEST_CASE("Signal View rejects unsupported input tensors",
          "[modules][signal_view][validation]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             DataType::CF32,
                                             {64});
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             DataType::F32,
                                             {});
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             DataType::F32,
                                             {2, 2, 2});

            Modules::SignalView decimated;
            decimated.decimation = 2;
            RequireSignalViewValidationError(implementation,
                                             decimated,
                                             DataType::F32,
                                             {3});
        }
    }
}

TEST_CASE("Signal View validates signal axis roles",
          "[modules][signal_view][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor missing(implementation.device, DataType::F32, {2, 32});
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             missing);

            Tensor malformed(implementation.device, DataType::F32, {32});
            REQUIRE(malformed.setAttribute(std::string(SampleAxisAttribute),
                                           I64{0}) == Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             malformed);

            Tensor mixed(implementation.device, DataType::F32, {2, 32});
            REQUIRE(SetSignalAxes(mixed, {
                .sample = Index{1},
                .channel = Index{0},
            }) == Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             mixed);

            Tensor auxiliary(implementation.device, DataType::F32, {2, 32});
            REQUIRE(SetSignalAxes(auxiliary, {
                .sample = Index{1},
            }) == Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             auxiliary);
        }
    }
}

TEST_CASE("Signal View rejects non-finite configuration and metadata",
          "[modules][signal_view][validation]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {64}) ==
                    Result::SUCCESS);

            Modules::SignalView config;
            config.rangeMin = -std::numeric_limits<F32>::infinity();
            RequireSignalViewValidationError(implementation, config, input);

            REQUIRE(input.setAttribute("sampleRate", F64{1.0e6}) ==
                    Result::SUCCESS);
            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             input);
        }
    }
}

TEST_CASE("Waterfall history tracks wrapped dirty rows",
          "[modules][signal_view][waterfall]") {
    constexpr U64 height = 5;
    Modules::WaterfallHistory history;

    history.advance(height, height);
    REQUIRE(history.writeIndex == 0);
    REQUIRE(history.dirtyRows == height);
    auto dirty = history.dirtyPlan(height);
    REQUIRE(dirty.startRow == 0);
    REQUIRE(dirty.firstRowCount == height);
    REQUIRE(dirty.secondRowCount == 0);

    history.clearDirty();
    history.advance(3, height);
    history.advance(4, height);
    REQUIRE(history.writeIndex == 2);
    REQUIRE(history.dirtyRows == height);
    dirty = history.dirtyPlan(height);
    REQUIRE(dirty.firstRowCount + dirty.secondRowCount == height);
}

TEST_CASE("Combined signal view keeps full-resolution waterfall rows",
          "[modules][signal_view][waterfall][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            constexpr U64 height = 3;
            constexpr U64 batches = 5;
            constexpr U64 rowWidth = 6;

            Tensor cpuInput;
            REQUIRE(cpuInput.create(DeviceType::CPU, DataType::F32,
                                    {batches, rowWidth}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::iota(cpuInput.data<F32>(),
                      cpuInput.data<F32>() + cpuInput.size(),
                      1.0f);

            Tensor input;
            if (implementation.device == DeviceType::CPU) {
                input = cpuInput;
            } else {
                REQUIRE(input.create(implementation.device, cpuInput) ==
                        Result::SUCCESS);
            }

            TensorMap inputs;
            inputs["signal"].requested("source", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.decimation = 2;
            config.waterfallHeight = height;
            REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const std::vector<F32> expected = {
                19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
            };
            REQUIRE(ReadWaterfallBins(module) == expected);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == 2);
            REQUIRE(ReadWaterfallHistory(module).dirtyRows == height);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View indexes sample and channel batch layouts equivalently",
           "[modules][signal_view][layout]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            for (const bool useChannelAxis : {false, true}) {
                DYNAMIC_SECTION("Element axis: "
                                << (useChannelAxis ? "channel" : "sample")) {
                    Tensor leading(DeviceType::CPU, DataType::F32, {2, 3});
                    const F32 leadingData[] = {
                        0.25f, 0.50f, 0.75f,
                        0.25f, 0.75f, 0.50f,
                    };
                    std::copy(std::begin(leadingData), std::end(leadingData),
                              leading.data<F32>());
                    SignalAxes leadingAxes{.batch = Index{0}};
                    if (useChannelAxis) {
                        leadingAxes.channel = Index{1};
                    } else {
                        leadingAxes.sample = Index{1};
                    }
                    REQUIRE(SetSignalAxes(leading, leadingAxes) == Result::SUCCESS);

                    Tensor trailing(DeviceType::CPU, DataType::F32, {3, 2});
                    const F32 trailingData[] = {
                        0.25f, 0.25f,
                        0.50f, 0.75f,
                        0.75f, 0.50f,
                    };
                    std::copy(std::begin(trailingData), std::end(trailingData),
                              trailing.data<F32>());
                    SignalAxes trailingAxes{.batch = Index{1}};
                    if (useChannelAxis) {
                        trailingAxes.channel = Index{0};
                    } else {
                        trailingAxes.sample = Index{0};
                    }
                    REQUIRE(SetSignalAxes(trailing, trailingAxes) == Result::SUCCESS);

                    const auto leadingSnapshot =
                        ComputeSignalViewSnapshot(implementation, leading);
                    const auto trailingSnapshot =
                        ComputeSignalViewSnapshot(implementation, trailing);
                    REQUIRE(trailingSnapshot.signalPoints ==
                            leadingSnapshot.signalPoints);
                    REQUIRE(trailingSnapshot.waterfallBins ==
                            leadingSnapshot.waterfallBins);
                }
            }
        }
    }
}

TEST_CASE("Signal View preserves waterfall history across runtime rebuilds",
          "[modules][signal_view][waterfall][runtime]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            constexpr U64 height = 5;
            constexpr U64 batches = 2 * height + 2;
            constexpr U64 width = 3;

            Tensor cpuInput(DeviceType::CPU, DataType::F32, {batches, width});
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::iota(cpuInput.data<F32>(),
                      cpuInput.data<F32>() + cpuInput.size(),
                      1.0f);

            Tensor input;
            if (implementation.device == DeviceType::CPU) {
                input = cpuInput;
            } else {
                REQUIRE(input.create(implementation.device, cpuInput) ==
                        Result::SUCCESS);
            }

            TensorMap inputs;
            inputs["signal"].requested("source", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallHeight = height;
            REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::vector<F32> expected(height * width, 0.0f);
            U64 expectedWriteIndex = 0;
            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            ApplyReferenceRows(expected, expectedWriteIndex, cpuInput, height);
            REQUIRE(ReadWaterfallBins(module) == expected);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == expectedWriteIndex);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            ApplyReferenceRows(expected, expectedWriteIndex, cpuInput, height);
            REQUIRE(ReadWaterfallBins(module) == expected);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == expectedWriteIndex);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View reconfigure preserves applied waterfall state",
          "[modules][signal_view][waterfall][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallHeight = 4;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            Parser::Map recreate;
            recreate["waterfallHeight"] = U64{8};
            REQUIRE(module->reconfigure(recreate, true) == Result::SUCCESS);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            REQUIRE(module->reconfigure(recreate) == Result::RECREATE);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto& applied =
                static_cast<const Modules::SignalView&>(module->config());
            REQUIRE(applied.waterfallHeight == config.waterfallHeight);

            Parser::Map rejected;
            rejected["waterfallHeight"] = U64{0};
            REQUIRE(module->reconfigure(rejected) == Result::ERROR);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(applied.waterfallHeight == config.waterfallHeight);
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View configuration omits fixed rendering settings",
          "[modules][signal_view][config]") {
    Modules::SignalView config;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(serialized.contains("interpolate"));
    REQUIRE_FALSE(serialized.contains("waterfallInterpolate"));
    REQUIRE_FALSE(serialized.contains("thickness"));
}

TEST_CASE("Signal View serializes plot labels", "[modules][signal_view][config]") {
    Modules::SignalView config;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE(std::any_cast<bool>(serialized.at("fill")));
    REQUIRE(std::any_cast<std::string>(serialized.at("xLabel")) ==
            "Frequency (MHz)");
    REQUIRE(std::any_cast<std::string>(serialized.at("amplitudeLabel")) ==
            "Amplitude (dBFS)");
    REQUIRE(std::any_cast<std::string>(serialized.at("waterfallLabel")) == "Time");

    config.xLabel = "";
    config.fill = false;
    config.amplitudeLabel = "Power";
    config.waterfallLabel = "History";
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(std::any_cast<bool>(serialized.at("fill")));
    REQUIRE(serialized.contains("xLabel"));
    REQUIRE(std::any_cast<std::string>(serialized.at("xLabel")).empty());
    REQUIRE(std::any_cast<std::string>(serialized.at("amplitudeLabel")) == "Power");
    REQUIRE(std::any_cast<std::string>(serialized.at("waterfallLabel")) == "History");
}

TEST_CASE("Lineplot helpers preserve indexing and max-hold warmup",
           "[modules][signal_view][lineplot][regression]") {
    REQUIRE(Modules::detail::LineplotInputIndex(1, 1, 5, 1, 2) == 7);
    REQUIRE_FALSE(Modules::detail::LineplotMaxHoldReady(2, 4));
    REQUIRE(Modules::detail::LineplotMaxHoldReady(3, 4));
    REQUIRE(Modules::detail::LineplotMaxHoldReady(0, 1));

    std::array<F32, 6> signalPoints;
    std::array<F32, 6> maxHoldPoints;
    Modules::detail::InitializeLineplotPoints(signalPoints.data(),
                                              maxHoldPoints.data(), 3);
    REQUIRE(signalPoints == std::array<F32, 6>{-1.0f, 0.0f, 0.0f, 0.0f,
                                               1.0f, 0.0f});
    REQUIRE(maxHoldPoints == std::array<F32, 6>{-1.0f, -1.0f, 0.0f, -1.0f,
                                                1.0f, -1.0f});
}

TEST_CASE("Signal View clamps amplitudes before averaging",
          "[modules][signal_view][lineplot][averaging][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor cpuInput(DeviceType::CPU, DataType::F32, {2, 4});
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(cpuInput.data<F32>(), cpuInput.size(),
                        -std::numeric_limits<F32>::infinity());

            Tensor input;
            if (implementation.device == DeviceType::CPU) {
                input = cpuInput;
            } else {
                REQUIRE(input.create(implementation.device, cpuInput) ==
                        Result::SUCCESS);
            }

            TensorMap inputs;
            inputs["signal"].requested("source", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 2;
            config.fill = false;
            REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto firstPoints = ReadSignalPoints(module);
            for (const F32 point : firstPoints) {
                REQUIRE(std::isfinite(point));
            }

            std::fill_n(cpuInput.data<F32>(), cpuInput.size(), 2.0f);
            if (implementation.device != DeviceType::CPU) {
                REQUIRE(input.copyFrom(cpuInput) == Result::SUCCESS);
            }
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto recoveredPoints = ReadSignalPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(std::isfinite(recoveredPoints[(index * 2) + 1]));
                REQUIRE(recoveredPoints[(index * 2) + 1] <= 1.0f);
                REQUIRE(recoveredPoints[(index * 2) + 1] >
                        firstPoints[(index * 2) + 1]);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View clears lineplot history on range reconfiguration",
          "[modules][signal_view][lineplot][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 4;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 8; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }

            const auto settledHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(settledHold[(index * 2) + 1] > -1.0f);
            }

            Parser::Map range;
            range["rangeMin"] = F32{-50.0f};
            range["rangeMax"] = F32{50.0f};
            REQUIRE(module->reconfigure(range) == Result::SUCCESS);

            const auto* impl = module->getImpl<Modules::SignalViewImpl>();
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == 0);
            const auto resetHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(resetHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const auto freshPoints = ReadSignalPoints(module);
            const auto freshHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(freshPoints[(index * 2) + 1] == -0.125f);
                REQUIRE(freshHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View clears lineplot history on averaging reconfiguration",
          "[modules][signal_view][lineplot][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 4;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 8; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }

            Parser::Map averaging;
            averaging["averaging"] = U64{16};
            REQUIRE(module->reconfigure(averaging) == Result::SUCCESS);

            const auto* impl = module->getImpl<Modules::SignalViewImpl>();
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == 0);
            const auto resetHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(resetHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const auto freshPoints = ReadSignalPoints(module);
            const auto freshHold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(freshPoints[(index * 2) + 1] == -0.03125f);
                REQUIRE(freshHold[(index * 2) + 1] == -1.0f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View preserves waterfall history on averaging reconfiguration",
          "[modules][signal_view][waterfall][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.5f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.averaging = 4;
            config.maxHold = true;
            config.waterfallHeight = 4;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 3; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }
            REQUIRE(ReadWaterfallHistory(module).writeIndex == 2);

            Parser::Map averaging;
            averaging["averaging"] = U64{16};
            REQUIRE(module->reconfigure(averaging) == Result::SUCCESS);

            const auto* impl = module->getImpl<Modules::SignalViewImpl>();
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == 0);
            const auto& history = ReadWaterfallHistory(module);
            REQUIRE(history.writeIndex == 2);

            const auto preservedBins = ReadWaterfallBins(module);
            for (const F32 value : preservedBins) {
                REQUIRE(value == 0.5f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View clears waterfall history on range reconfiguration",
          "[modules][signal_view][waterfall][reconfigure][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.5f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallHeight = 4;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 3; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }
            REQUIRE(ReadWaterfallHistory(module).writeIndex == 2);

            Parser::Map range;
            range["rangeMin"] = F32{-50.0f};
            range["rangeMax"] = F32{50.0f};
            REQUIRE(module->reconfigure(range) == Result::SUCCESS);

            const auto& history = ReadWaterfallHistory(module);
            REQUIRE(history.writeIndex == 0);
            REQUIRE(history.dirtyRows == 4);
            const auto resetBins = ReadWaterfallBins(module);
            for (const F32 value : resetBins) {
                REQUIRE(value == 0.0f);
            }

            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const auto freshBins = ReadWaterfallBins(module);
            for (U64 row = 0; row < 4; ++row) {
                for (U64 sample = 0; sample < 8; ++sample) {
                    REQUIRE(freshBins[(row * 8) + sample] ==
                            (row < 2 ? 0.5f : 0.0f));
                }
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View max hold captures the first observation with averaging one",
          "[modules][signal_view][lineplot][maxhold][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 1;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            const auto hold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(hold[(index * 2) + 1] == -0.5f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View max hold captures the first fully averaged observation",
          "[modules][signal_view][lineplot][maxhold][regression]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {1, 8}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(input.data<F32>(), input.size(), 0.25f);

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            Modules::SignalView config;
            config.mode = "lineplot";
            config.averaging = 4;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            for (U64 i = 0; i < 4; i++) {
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            }

            const auto hold = ReadMaxHoldPoints(module);
            for (U64 index = 0; index < input.shape(1); ++index) {
                REQUIRE(hold[(index * 2) + 1] == -0.341796875f);
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
