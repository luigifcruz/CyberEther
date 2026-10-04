#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "jetstream/domains/visualization/signal_view/module.hh"
#include "jetstream/domains/core/range/module.hh"
#include "jetstream/domains/dsp/amplitude/module.hh"
#include "jetstream/domains/dsp/decimator/block.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/runtime.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/testing.hh"
#include "flowgraph_fixture.hh"

#include "module_impl.hh"
#include "waterfall_3d.hh"

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

    static Modules::SignalViewCanvas& canvasOf(Modules::SignalViewImpl& impl) {
        return impl.*&SignalViewImplAccess::canvas;
    }

    static Modules::SignalViewLineplot& lineplotOf(Modules::SignalViewImpl& impl) {
        return impl.*&SignalViewImplAccess::lineplot;
    }

    static Modules::SignalViewWaterfall& waterfallOf(Modules::SignalViewImpl& impl) {
        return impl.*&SignalViewImplAccess::waterfall;
    }

    static auto configChangePendingMember() {
        return &SignalViewImplAccess::configChangePending;
    }

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
    static void wirePresentResources(Modules::SignalViewImpl& impl,
                                     const std::shared_ptr<Render::Components::Axis>& axis,
                                     const std::shared_ptr<Render::Components::Text>& text,
                                     const std::shared_ptr<Render::Components::Text>& cursorText = {},
                                     const std::shared_ptr<Render::Components::Text>& markerText = {});
#endif
};

struct InteractiveSignalViewConfig : Block::Config {
    F32 splitRatio = 0.5f;
    std::vector<F32> markers;
    std::vector<U64> pins;
    JST_BLOCK_TYPE(interactive_signal_view_test);
    JST_BLOCK_DOMAIN("Test");
    JST_BLOCK_PARAMS(splitRatio, markers, pins);
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
        config->markers = markers;
        config->pins = pins;
        return Result::SUCCESS;
    }
    Result create() override {
        JST_CHECK(moduleCreate("plot", config, inputs()));
        JST_CHECK(moduleBindConfigEdit("plot", "splitRatio", "splitRatio"));
        JST_CHECK(moduleBindConfigEdit("plot", "markers", "markers"));
        JST_CHECK(moduleBindConfigEdit("plot", "pins", "pins"));
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

class LabelTestText final : public Render::Components::Text {
 public:
    explicit LabelTestText(const Config& config) : Text(config) {}

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
    explicit LabelTestBuffer(const Config& config = {}) : Buffer(config) {}

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
};

void SignalViewImplAccess::wirePresentResources(
    Modules::SignalViewImpl& impl,
    const std::shared_ptr<Render::Components::Axis>& axis,
    const std::shared_ptr<Render::Components::Text>& text,
    const std::shared_ptr<Render::Components::Text>& cursorText,
    const std::shared_ptr<Render::Components::Text>& markerText) {
    // The combined presentation path updates real CPU state and queues uploads,
    // but these resources never bind to a GPU. Only data members are accessed;
    // internal non-exported implementation methods must not be called by tests.
    auto& canvas = SignalViewImplAccess::canvasOf(impl);
    auto& lineplot = SignalViewImplAccess::lineplotOf(impl);
    auto& waterfall = SignalViewImplAccess::waterfallOf(impl);
    canvas.axis = axis;
    canvas.cursorText = cursorText;
    canvas.text = text;
    canvas.markerText = markerText;
    canvas.renderSurface = std::make_shared<LabelTestSurface>();
    const auto tensorBuffer = [](Tensor& tensor) {
        return std::make_shared<LabelTestBuffer>(Render::Buffer::Config{
            .size = tensor.size(),
            .target = Render::Buffer::Target::STORAGE,
            .elementByteSize = sizeof(F32),
            .buffer = tensor.data(),
        });
    };
    if (impl.*&SignalViewImplAccess::lineplotEnabled) {
        lineplot.signalPointsBuffer = tensorBuffer(impl.*signalPointsMember());
        lineplot.maxHoldPointsBuffer = tensorBuffer(impl.*maxHoldPointsMember());
    }
    if (impl.*&SignalViewImplAccess::waterfallEnabled) {
        waterfall.upload.resize((impl.*&SignalViewImplAccess::numberOfElements + 16) *
                                (impl.*&SignalViewImplAccess::waterfallHeight));
        waterfall.binsBuffer = std::make_shared<LabelTestBuffer>(Render::Buffer::Config{
            .size = waterfall.upload.size(), .target = Render::Buffer::Target::STORAGE,
            .elementByteSize = sizeof(F32), .buffer = waterfall.upload.data(),
        });
    }
    lineplot.signalUniformBuffer = std::make_shared<LabelTestBuffer>();
    lineplot.holdUniformBuffer = std::make_shared<LabelTestBuffer>();
    waterfall.uniformBuffer = std::make_shared<LabelTestBuffer>(
        Render::Buffer::Config{
            .size = 1,
            .target = Render::Buffer::Target::UNIFORM,
            .elementByteSize = sizeof(waterfall.uniforms),
            .buffer = &waterfall.uniforms,
        });
    lineplot.signalKernel = std::make_shared<Render::Kernel>(Render::Kernel::Config{});
    lineplot.fillKernel = std::make_shared<Render::Kernel>(Render::Kernel::Config{});
    lineplot.maxHoldKernel = std::make_shared<Render::Kernel>(Render::Kernel::Config{});
    lineplot.signalProgram = std::make_shared<Render::Program>(Render::Program::Config{});
    lineplot.fillProgram = std::make_shared<Render::Program>(Render::Program::Config{});
    lineplot.maxHoldProgram = std::make_shared<Render::Program>(Render::Program::Config{});
    waterfall.program = std::make_shared<Render::Program>(Render::Program::Config{});
}

std::shared_ptr<LabelTestText> MakeMarkerText(const std::shared_ptr<Render::Components::Font>& font,
                                              LabelTestWindow& window) {
    Render::Components::Text::Config config;
    config.font = font;
    config.maxCharacters = 1024;
    for (U64 i = 0; i < Modules::detail::MaxMarkers; ++i) {
        config.elements[jst::fmt::format("marker-{}-tag", i)] = {};
        config.elements[jst::fmt::format("marker-{}-id", i)] = {};
        config.elements[jst::fmt::format("marker-{}-x", i)] = {};
        config.elements[jst::fmt::format("marker-{}-y", i)] = {};
    }
    for (U64 i = 0; i < Modules::detail::MarkerSpans; ++i) {
        config.elements[jst::fmt::format("span-{}-label", i)] = {};
    }
    auto text = std::make_shared<LabelTestText>(config);
    REQUIRE(text->create(&window) == Result::SUCCESS);
    return text;
}

U64 ApplyBufferUploads(const std::shared_ptr<Render::Buffer>& buffer, void* destination) {
    if (!buffer) return 0;
    Render::Transfer::Batch batch;
    batch.collect(buffer);
    U64 bytes = 0;
    for (const auto& transfer : batch.buffers()) {
        REQUIRE(transfer.destinationOffset + transfer.upload.data.size() <= buffer->byteSize());
        std::memcpy(static_cast<U8*>(destination) + transfer.destinationOffset,
                    transfer.upload.data.data(), transfer.upload.data.size());
        bytes += transfer.upload.data.size();
    }
    batch.commit();
    return bytes;
}
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

void RequirePoints(const std::vector<F32>& actual, const std::vector<F32>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (U64 i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(1e-6f));
    }
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

TEST_CASE("Signal View amplitude labels invert the soft display mapping",
          "[modules][signal_view][lineplot][labels][numeric]") {
    const bool reversed = GENERATE(false, true);
    const F32 min = reversed ? 0.0f : -100.0f;
    const F32 max = reversed ? -100.0f : 0.0f;
    const F32 positions[] = {-0.99505475f, -0.96402758f, -0.76159416f,
                              0.0f, 0.76159416f, 0.96402758f, 0.99505475f};
    const char* labels[] = {"-125", "-100", "-75", "-50", "-25", "0", "25"};
    for (U64 i = 0; i < 7; ++i) {
        CAPTURE(positions[i], reversed);
        REQUIRE(Modules::detail::LineplotAmplitudeLabel(positions[i], min, max) == labels[i]);
    }
    REQUIRE(Modules::detail::LineplotAmplitudeLabel(0.0f, -200.0f, 0.0f) == "-100");
    REQUIRE(Modules::detail::LineplotAmplitudeLabel(0.5f, -50.0f, -50.0f) == "-50");
    for (const F32 position : {-1.0f, 1.0f, -2.0f, 2.0f,
                               std::numeric_limits<F32>::infinity(),
                               -std::numeric_limits<F32>::infinity(),
                               std::numeric_limits<F32>::quiet_NaN()}) {
        REQUIRE(Modules::detail::LineplotAmplitudeLabel(position, min, max).empty());
    }
}

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
TEST_CASE("Axis vertical scale can move the divider without recreating resources",
          "[modules][signal_view][split][axis][regression]") {
    LabelTestWindow window;
    Render::Components::Axis::Config config;
    config.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    LabelTestAxis axis(config);
    // CPU-side setup must not submit draw commands before a surface is bound.
    JST_LOG_LAST_ERROR().clear();
    REQUIRE(axis.create(&window) == Result::SUCCESS);
    CHECK(JST_LOG_LAST_ERROR() == "");
    REQUIRE(axis.updatePixelSize({2.0f / 1000.0f, 2.0f / 800.0f}) == Result::SUCCESS);
    CHECK(JST_LOG_LAST_ERROR() == "");
    for (const F32 ratio : {0.1f, 0.35f, 0.9f, 1.0f, 0.5f}) {
        REQUIRE(axis.updateVerticalScale(ratio) == Result::SUCCESS);
        CHECK(JST_LOG_LAST_ERROR() == "");
        REQUIRE(axis.getConfig().verticalScale == ratio);
        REQUIRE(axis.currentHorizontalLineCount() >= 3);
    }
    REQUIRE(axis.updateVerticalScale(0.0f) == Result::ERROR);
    REQUIRE(axis.updateVerticalScale(std::numeric_limits<F32>::quiet_NaN()) == Result::ERROR);
    REQUIRE(axis.getConfig().verticalScale == 0.5f);
    REQUIRE(axis.destroy(&window) == Result::SUCCESS);
}

TEST_CASE("Axis created in split mode reserves full height tick geometry",
          "[modules][signal_view][split][axis][capacity][regression]") {
    LabelTestWindow window;
    Render::Components::Axis::Config config;
    config.verticalScale = 0.5f;
    config.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    LabelTestAxis axis(config);
    JST_LOG_LAST_ERROR().clear();
    REQUIRE(axis.create(&window) == Result::SUCCESS);
    CHECK(JST_LOG_LAST_ERROR() == "");
    REQUIRE(axis.updatePixelSize({2.0f / 16384.0f, 2.0f / 16384.0f}) == Result::SUCCESS);
    CHECK(JST_LOG_LAST_ERROR() == "");
    REQUIRE(axis.currentVerticalLineCount() == 65);
    REQUIRE(axis.currentHorizontalLineCount() == 17);
    REQUIRE(axis.setShowFrameTicks(true) == Result::SUCCESS);
    CHECK(JST_LOG_LAST_ERROR() == "");

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
        CHECK(JST_LOG_LAST_ERROR() == "");
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
    SignalViewImplAccess::canvasOf(*impl).axis = axis;
    SignalViewImplAccess::canvasOf(*impl).renderSurface =
        std::make_shared<LabelTestSurface>();
    SignalViewImplAccess::waterfallOf(*impl).uniformBuffer =
        std::make_shared<LabelTestBuffer>();
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);

    // Ignore validation/creation reads. Present must reread live metadata even
    // without new samples, mouse events, or surface resize/placement events.
    const auto present = [&] {
        frequencyReads = 0;
        sampleRateReads = 0;
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        CHECK_FALSE(SignalViewImplAccess::canvasOf(*impl).interaction.viewChanged);
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

TEST_CASE("Signal View label units come from the trailing parentheses",
          "[modules][signal_view][labels][cursor]") {
    REQUIRE(Modules::detail::LabelUnit("Amplitude (dBFS)") == "dBFS");
    REQUIRE(Modules::detail::LabelUnit("Power (dBm) (avg)") == "avg");
    REQUIRE(Modules::detail::LabelUnit("Time") == "");
    REQUIRE(Modules::detail::LabelUnit("Empty ()") == "");
    REQUIRE(Modules::detail::LabelUnit(") mismatched (") == "");
}

TEST_CASE("Signal View marker spans format frequency distances by magnitude",
          "[modules][signal_view][markers]") {
    REQUIRE(Modules::detail::FormatFrequencySpan(9.8e6) == "9.8 MHz");
    REQUIRE(Modules::detail::FormatFrequencySpan(1.0e6) == "1.0 MHz");
    REQUIRE(Modules::detail::FormatFrequencySpan(1.2345678e6) == "1.235 MHz");
    REQUIRE(Modules::detail::FormatFrequencySpan(-12.5e3) == "12.5 kHz");
    REQUIRE(Modules::detail::FormatFrequencySpan(500.0e3) == "500.0 kHz");
    REQUIRE(Modules::detail::FormatFrequencySpan(250.0) == "250.0 Hz");
    REQUIRE(Modules::detail::FormatFrequencySpan(0.0) == "0.0 Hz");
}

TEST_CASE("Cursor readout follows the mouse over the plot and hides when it leaves",
          "[modules][signal_view][present][cursor]") {
    const std::string mode = GENERATE("lineplot", "waterfall", "lineplot_waterfall");
    const bool lineplot = Modules::detail::SignalViewHasLineplot(mode);
    CAPTURE(mode);

    Tensor input(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(SetSignalAxes(input, {.sample = Index{1}, .batch = Index{0}}) == Result::SUCCESS);
    REQUIRE(input.setDerivedAttribute("frequency", []() -> std::any {
        return 100.0e6f;
    }) == Result::SUCCESS);
    REQUIRE(input.setDerivedAttribute("sampleRate", []() -> std::any {
        return 2.0e6f;
    }) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].tensor = input;
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view", DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic", module) == Result::SUCCESS);
    Modules::SignalView config;
    config.mode = mode;
    config.waterfallHeight = 4;
    REQUIRE(module->create("plot", config, inputs) == Result::SUCCESS);
    Runtime runtime("plot", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"plot", module}}) == Result::SUCCESS);

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    Render::Components::Text::Config textConfig;
    textConfig.font = axisConfig.font;
    textConfig.maxCharacters = 256;
    textConfig.elements = {{"header", {}}, {"zoom", {}}, {"hold", {}},
                           {"amplitude-title", {}}, {"waterfall-title", {}}};
    auto text = std::make_shared<LabelTestText>(textConfig);
    REQUIRE(text->create(&window) == Result::SUCCESS);
    Render::Components::Text::Config cursorConfig;
    cursorConfig.font = axisConfig.font;
    cursorConfig.maxCharacters = 64;
    cursorConfig.elements = {{"cursor-x", {}}, {"cursor-y", {}}};
    auto cursorText = std::make_shared<LabelTestText>(cursorConfig);
    REQUIRE(cursorText->create(&window) == Result::SUCCESS);
    auto markerText = MakeMarkerText(axisConfig.font, window);
    auto* impl = module->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    SignalViewImplAccess::wirePresentResources(*impl, axis, text, cursorText, markerText);
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);

    std::unordered_set<std::string> skipped, failed;
    std::fill_n(input.data<F32>(), input.size(), 0.4f);
    REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == " ");
    REQUIRE(cursorText->get("cursor-y").fill == " ");

    const auto amplitudeAt = [&](const U64 bin) -> std::string {
        if (!lineplot) return " ";
        const auto value = Modules::detail::LineplotAmplitudeValue(
            ReadSignalPoints(module)[(bin * 2) + 1], config.rangeMin, config.rangeMax);
        REQUIRE(value.has_value());
        return jst::fmt::format("{:.1f} dBFS", *value);
    };

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == "100.0000 MHz");
    REQUIRE(cursorText->get("cursor-y").fill == amplitudeAt(4));

    const auto padding = axis->paddingScale();
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f + 0.25f * padding.x, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == "100.5000 MHz");
    REQUIRE(cursorText->get("cursor-y").fill == amplitudeAt(5));

    const auto markerLabel = [&](const U64 index, const char* suffix) {
        return markerText->get(jst::fmt::format("marker-{}-{}", index, suffix)).fill;
    };
    const auto& interaction = SignalViewImplAccess::canvasOf(*impl).interaction;
    const auto shiftClick = [&](const Extent2D<F32>& position, const bool clear = false) {
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Click,
            .button = clear ? MouseButton::Right : MouseButton::Left,
            .position = position,
            .modifiers = {.shift = true},
        });
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        REQUIRE_FALSE(interaction.dragging);
        REQUIRE(interaction.zoom == Catch::Approx(1.0f));
    };
    const Extent2D<F32> quarter = {0.5f + 0.25f * padding.x, 0.5f};
    REQUIRE(markerLabel(0, "x") == " ");
    shiftClick(quarter);
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions == std::vector<F32>{0.5f});
    REQUIRE(markerLabel(0, "tag") == "M1");
    REQUIRE(markerLabel(0, "id") == "M1");
    REQUIRE(markerLabel(0, "x") == "100.5000 MHz");
    REQUIRE(markerLabel(0, "y") == amplitudeAt(5));
    REQUIRE(markerLabel(1, "x") == " ");
    REQUIRE(markerLabel(1, "id") == " ");
    shiftClick(quarter);
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions.empty());
    REQUIRE(markerLabel(0, "x") == " ");
    REQUIRE(markerLabel(0, "y") == " ");
    shiftClick(quarter);
    shiftClick({0.5f, 0.5f});
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions == std::vector<F32>{0.5f, 0.0f});
    REQUIRE(markerLabel(0, "x") == "100.5000 MHz");
    REQUIRE(markerLabel(1, "id") == "M2");
    REQUIRE(markerLabel(1, "x") == "100.0000 MHz");
    REQUIRE(markerLabel(1, "y") == amplitudeAt(4));

    const auto spanLabel = [&](const U64 index, const char* suffix) {
        return markerText->get(jst::fmt::format("span-{}-{}", index, suffix)).fill;
    };
    const auto hoverTag = [&](const U64 index) {
        const auto tag = markerText->get(jst::fmt::format("marker-{}-tag", index)).position;
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Move,
            .position = {(tag.x + 1.0f) * 0.5f, (1.0f - tag.y) * 0.5f},
        });
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    };
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == " ");
    hoverTag(1);
    REQUIRE(cursorText->get("cursor-x").fill == " ");
    REQUIRE(spanLabel(0, "label") == "500.0 kHz");
    REQUIRE(spanLabel(1, "label") == " ");
    const Extent2D<F32> lowerQuarter = {0.5f - 0.25f * padding.x, 0.5f};
    shiftClick(lowerQuarter);
    REQUIRE(markerLabel(2, "x") == "99.5000 MHz");
    hoverTag(1);
    REQUIRE(spanLabel(0, "label") == "500.0 kHz");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    hoverTag(0);
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    hoverTag(2);
    REQUIRE(spanLabel(0, "label") == "500.0 kHz");
    REQUIRE(spanLabel(1, "label") == " ");
    const auto moveAway = [&]() {
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Move, .position = {0.5f - 0.05f * padding.x, 0.5f},
        });
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    };
    moveAway();
    REQUIRE(cursorText->get("cursor-x").fill == "99.9000 MHz");
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == " ");
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == " ");
    REQUIRE(cursorText->get("cursor-y").fill == " ");
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == " ");

    const auto clickTag = [&](const U64 index) {
        const auto tag = markerText->get(jst::fmt::format("marker-{}-tag", index)).position;
        const Extent2D<F32> position = {(tag.x + 1.0f) * 0.5f, (1.0f - tag.y) * 0.5f};
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Click, .button = MouseButton::Left, .position = position,
        });
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Release, .button = MouseButton::Left, .position = position,
        });
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        REQUIRE_FALSE(interaction.dragging);
    };
    clickTag(0);
    moveAway();
    REQUIRE(cursorText->get("cursor-x").fill == "99.9000 MHz");
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    hoverTag(2);
    REQUIRE(spanLabel(0, "label") == "500.0 kHz");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    hoverTag(1);
    REQUIRE(spanLabel(0, "label") == "500.0 kHz");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    moveAway();
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    clickTag(0);
    moveAway();
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == " ");
    clickTag(0);
    clickTag(2);
    moveAway();
    REQUIRE(spanLabel(0, "label") == "500.0 kHz");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Escape, {}});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == " ");
    clickTag(0);
    clickTag(2);
    moveAway();
    clickTag(2);
    moveAway();
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == "500.0 kHz");
    clickTag(2);
    shiftClick(lowerQuarter);
    REQUIRE(spanLabel(0, "label") == "500.0 kHz");
    REQUIRE(spanLabel(1, "label") == " ");
    clickTag(0);
    moveAway();
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == " ");
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions == std::vector<F32>{0.5f, 0.0f});

    const auto dragMarker = [&](const Extent2D<F32>& from, const Extent2D<F32>& to) {
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Click, .button = MouseButton::Left, .position = from,
        });
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Move, .position = to,
        });
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Release, .button = MouseButton::Left, .position = to,
        });
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        REQUIRE_FALSE(interaction.dragging);
        REQUIRE(interaction.zoom == Catch::Approx(1.0f));
        REQUIRE(interaction.offset == Catch::Approx(0.0f));
    };
    dragMarker({0.5f, 0.5f}, {0.5f + 0.125f * padding.x, 0.5f});
    REQUIRE(markerLabel(1, "x") == "100.2500 MHz");
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions[1] == Catch::Approx(0.25f));
    const auto tagOrigin = markerText->get("marker-0-tag").position;
    const Extent2D<F32> tagFrom = {(tagOrigin.x + 1.0f) * 0.5f, (1.0f - tagOrigin.y) * 0.5f};
    dragMarker(tagFrom, {0.5f + 0.375f * padding.x, tagFrom.y});
    REQUIRE(markerLabel(0, "x") == "100.5000 MHz");
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions[0] == Catch::Approx(0.5f));
    moveAway();
    REQUIRE(spanLabel(0, "label") == "250.0 kHz");
    REQUIRE(spanLabel(1, "label") == " ");
    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Escape, {}});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(spanLabel(0, "label") == " ");
    REQUIRE(spanLabel(1, "label") == " ");
    for (U64 extra = 1; extra <= 16; ++extra) {
        shiftClick({0.5f - 0.03f * extra * padding.x, 0.5f});
    }
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions.size() == Modules::detail::MaxMarkers);
    REQUIRE(markerLabel(15, "id") == "M16");
    REQUIRE(markerLabel(15, "x") == "99.1600 MHz");
    shiftClick({0.5f, 0.5f}, true);
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions.empty());
    for (U64 i = 0; i < Modules::detail::MaxMarkers; ++i) {
        REQUIRE(markerLabel(i, "x") == " ");
    }
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f + 0.25f * padding.x, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);

    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Space, {}});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    const auto heldAmplitude = cursorText->get("cursor-y").fill;
    REQUIRE(heldAmplitude == amplitudeAt(5));
    std::fill_n(input.data<F32>(), input.size(), 0.9f);
    REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-y").fill == heldAmplitude);
    if (lineplot) REQUIRE(heldAmplitude != amplitudeAt(5));
    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Space, {}});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-y").fill == amplitudeAt(5));

    shiftClick(quarter);
    REQUIRE(markerLabel(0, "tag") == "M1");
    REQUIRE(markerLabel(0, "x") == "100.5000 MHz");
    module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize,
                                         .size = {512, 512},
                                         .placement = SurfacePlacementType::Attached});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == " ");
    REQUIRE(cursorText->get("cursor-y").fill == " ");
    REQUIRE(markerLabel(0, "tag") == "M1");
    REQUIRE(markerLabel(0, "id") == " ");
    REQUIRE(markerLabel(0, "x") == " ");
    REQUIRE(markerLabel(0, "y") == " ");

    module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize,
                                         .size = {512, 512},
                                         .placement = SurfacePlacementType::Detached});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == " ");
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == "100.0000 MHz");
    REQUIRE(markerLabel(0, "id") == "M1");
    REQUIRE(markerLabel(0, "x") == "100.5000 MHz");

    module->surface()->pushInputEvent(FocusEvent{false});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursorText->get("cursor-x").fill == " ");
    REQUIRE(cursorText->get("cursor-y").fill == " ");

    REQUIRE(markerText->destroy(&window) == Result::SUCCESS);
    REQUIRE(cursorText->destroy(&window) == Result::SUCCESS);
    REQUIRE(text->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Standalone marker displays survive resizing and follow reconfiguration",
          "[modules][signal_view][markers][present][reconfigure][regression]") {
    const std::string mode = GENERATE("lineplot", "waterfall", "lineplot_waterfall");
    CAPTURE(mode);
    Tensor input(DeviceType::CPU, DataType::F32, {8});
    TensorMap inputs;
    inputs["signal"].tensor = input;
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view", DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic", module) == Result::SUCCESS);
    Modules::SignalView config;
    config.mode = mode;
    config.waterfallHeight = 4;
    config.markers = {0.5f};
    REQUIRE(module->create("plot", config, inputs) == Result::SUCCESS);

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    auto markerText = MakeMarkerText(axisConfig.font, window);
    auto* impl = module->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    SignalViewImplAccess::wirePresentResources(*impl, axis, {}, {}, markerText);
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);
    const auto present = [&] { REQUIRE(presenter->presentSubmit() == Result::SUCCESS); };
    const auto& positions = SignalViewImplAccess::canvasOf(*impl).markerPositions;
    present();
    REQUIRE(markerText->get("marker-0-tag").fill == "M1");
    REQUIRE(markerText->get("marker-0-tag").position.x > 0.0f);

    SECTION("an existing marker survives narrow surfaces and recovers after expansion") {
        const U64 scale = GENERATE(1, 2);
        for (const U64 width : {88, 64, 512}) {
            CAPTURE(width, scale);
            module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize,
                                                 .size = {width * scale, 512 * scale},
                                                 .scale = static_cast<F32>(scale)});
            present();
            REQUIRE(positions == std::vector<F32>{0.5f});
            const auto tag = markerText->get("marker-0-tag");
            REQUIRE(std::isfinite(tag.position.x));
            REQUIRE(std::isfinite(tag.position.y));
            if (width == 64) {
                REQUIRE(tag.fill == " ");
            } else {
                REQUIRE(tag.fill == "M1");
                if (width <= 88) {
                    REQUIRE(tag.position.x == 0.0f);
                } else {
                    REQUIRE(tag.position.x > 0.0f);
                    REQUIRE(tag.position.x <= axis->paddingScale().x);
                }
            }
        }
    }

    SECTION("external edits update displayed markers and defer during a drag") {
        Parser::Map edit;
        edit["markers"] = std::vector<F32>{-0.5f, 0.25f};
        REQUIRE(module->reconfigure(edit, true) == Result::SUCCESS);
        present();
        REQUIRE(positions == std::vector<F32>{0.5f});
        REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
        present();
        REQUIRE(positions == std::vector<F32>{-0.5f, 0.25f});
        REQUIRE(markerText->get("marker-1-tag").fill == "M2");
        REQUIRE(markerText->get("marker-0-x").fill == (mode == "waterfall" ? "0.2500" : "-0.5000"));
        REQUIRE(markerText->get("marker-1-x").fill == (mode == "waterfall" ? "0.6250" : "0.2500"));

        const auto padding = axis->paddingScale();
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Click, .button = MouseButton::Left,
            .position = {0.5f - 0.25f * padding.x, 0.5f},
        });
        present();
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Move, .position = {0.5f, 0.5f},
        });
        present();
        REQUIRE(positions == std::vector<F32>{0.0f, 0.25f});
        edit["markers"] = std::vector<F32>{0.75f};
        REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
        for (U64 frame = 0; frame < 2; ++frame) {
            present();
            REQUIRE(positions == std::vector<F32>{0.0f, 0.25f});
            REQUIRE(markerText->get("marker-0-x").fill == (mode == "waterfall" ? "0.5000" : "0.0000"));
        }
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Release, .button = MouseButton::Left, .position = {0.5f, 0.5f},
        });
        present();
        present();
        REQUIRE(positions == std::vector<F32>{0.75f});
        REQUIRE(markerText->get("marker-0-x").fill == (mode == "waterfall" ? "0.8750" : "0.7500"));
        REQUIRE(markerText->get("marker-1-tag").fill == " ");

        edit["markers"] = std::vector<F32>{};
        REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
        present();
        REQUIRE(positions.empty());
        REQUIRE(markerText->get("marker-0-tag").fill == " ");
        REQUIRE(markerText->get("marker-0-x").fill == " ");
    }

    REQUIRE(markerText->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture, "Production plot markers persist after recreation and flowgraph reload",
                 "[modules][signal_view][markers][config-edits][regression]") {
    const std::string type = GENERATE("lineplot", "waterfall");
    CAPTURE(type);
    // Observe the real module factory through a test provider, so the production
    // block's configuration forwarding and edit bindings are exercised intact.
    struct ModuleObserver {
        const ProviderType provider = "marker-persistence-test";
        std::shared_ptr<Module> module;

        ModuleObserver() {
            const auto registrations = Registry::ListAvailableModules(
                "signal_view", DeviceType::CPU, RuntimeType::NATIVE, "generic");
            REQUIRE(registrations.size() == 1);
            REQUIRE(Registry::RegisterModule("signal_view", DeviceType::CPU, RuntimeType::NATIVE,
                provider, [this, factory = registrations.front().factory](const auto& environment, const auto& view) {
                    module = factory(environment, view);
                    return module;
                }) == Result::SUCCESS);
        }

        ~ModuleObserver() {
            CHECK(Registry::UnregisterModule("signal_view", DeviceType::CPU,
                                              RuntimeType::NATIVE, provider) == Result::SUCCESS);
        }
    } observer;

    TestFlowgraph::SyntheticSourceBlockConfig source;
    source.bufferSize = 32;
    REQUIRE(flowgraph->blockCreate("source", source, {}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].requested("source", "signal");
    REQUIRE(flowgraph->blockCreate("plot", type, {}, inputs, DeviceType::CPU,
                                   RuntimeType::NATIVE, observer.provider) == Result::SUCCESS);
    REQUIRE(viewBlock("plot").state == Block::State::Created);
    REQUIRE(observer.module);
    const auto original = observer.module;

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    auto markerText = MakeMarkerText(axisConfig.font, window);
    const auto wire = [&] {
        auto* impl = observer.module->getImpl<Modules::SignalViewImpl>();
        REQUIRE(impl);
        SignalViewImplAccess::wirePresentResources(*impl, axis, {}, {}, markerText);
    };
    const auto present = [&] {
        REQUIRE(observer.module->getImpl<Scheduler::Context>()->presentSubmit() == Result::SUCCESS);
    };
    const auto stored = [&](Flowgraph& graph) {
        return std::any_cast<std::vector<F32>>(ViewBlock(graph, "plot").config.at("markers"));
    };
    wire();
    present();
    original->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Click, .button = MouseButton::Left,
        .position = {0.5f + 0.25f * axis->paddingScale().x, 0.5f},
        .modifiers = {.shift = true},
    });
    present();
    present();
    REQUIRE(markerText->get("marker-0-tag").fill == "M1");
    REQUIRE(stored(*flowgraph).empty());
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE(stored(*flowgraph) == std::vector<F32>{0.5f});

    REQUIRE(flowgraph->blockRecreate("plot", viewBlock("plot").config) == Result::SUCCESS);
    REQUIRE(observer.module != original);
    wire();
    present();
    REQUIRE(markerText->get("marker-0-tag").fill == "M1");
    REQUIRE(markerText->get("marker-0-x").fill == (type == "waterfall" ? "0.7500" : "0.5000"));

    std::vector<char> blob;
    REQUIRE(flowgraph->exportToBlob(blob) == Result::SUCCESS);
    Flowgraph restored;
    REQUIRE(restored.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(restored.importFromBlob(blob) == Result::SUCCESS);
    REQUIRE(stored(restored) == std::vector<F32>{0.5f});
    wire();
    present();
    REQUIRE(markerText->get("marker-0-tag").fill == "M1");
    REQUIRE(markerText->get("marker-0-x").fill == (type == "waterfall" ? "0.7500" : "0.5000"));
    REQUIRE(restored.destroy() == Result::SUCCESS);
    REQUIRE(markerText->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
}

TEST_CASE("Cursor stays inside the zoomed plot and interpolates between sparse samples",
          "[modules][signal_view][present][cursor][zoom][regression]") {
    Tensor input(DeviceType::CPU, DataType::F32, {2, 2});
    REQUIRE(SetSignalAxes(input, {.sample = Index{1}, .batch = Index{0}}) == Result::SUCCESS);
    REQUIRE(input.setDerivedAttribute("frequency", []() -> std::any {
        return 100.0e6f;
    }) == Result::SUCCESS);
    REQUIRE(input.setDerivedAttribute("sampleRate", []() -> std::any {
        return 2.0e6f;
    }) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].tensor = input;
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view", DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic", module) == Result::SUCCESS);
    Modules::SignalView config;
    config.mode = "lineplot";
    REQUIRE(module->create("plot", config, inputs) == Result::SUCCESS);
    Runtime runtime("plot", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"plot", module}}) == Result::SUCCESS);

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    Render::Components::Text::Config textConfig;
    textConfig.font = axisConfig.font;
    textConfig.maxCharacters = 256;
    textConfig.elements = {{"header", {}}, {"zoom", {}}, {"hold", {}},
                           {"amplitude-title", {}}, {"waterfall-title", {}}};
    auto text = std::make_shared<LabelTestText>(textConfig);
    REQUIRE(text->create(&window) == Result::SUCCESS);
    Render::Components::Text::Config cursorConfig;
    cursorConfig.font = axisConfig.font;
    cursorConfig.maxCharacters = 64;
    cursorConfig.elements = {{"cursor-x", {}}, {"cursor-y", {}}};
    auto cursorText = std::make_shared<LabelTestText>(cursorConfig);
    REQUIRE(cursorText->create(&window) == Result::SUCCESS);
    auto* impl = module->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    SignalViewImplAccess::wirePresentResources(*impl, axis, text, cursorText);
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);
    const auto& cursor = SignalViewImplAccess::canvasOf(*impl).cursor;
    const auto& interaction = SignalViewImplAccess::canvasOf(*impl).interaction;

    std::unordered_set<std::string> skipped, failed;
    for (U64 b = 0; b < 2; ++b) {
        input.data<F32>()[b * 2 + 0] = 0.25f;
        input.data<F32>()[b * 2 + 1] = 0.75f;
    }
    REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    const auto points = ReadSignalPoints(module);
    REQUIRE(points[1] == Catch::Approx(std::tanh(-1.0f)));
    REQUIRE(points[3] == Catch::Approx(std::tanh(1.0f)));

    const auto expectedAmplitude = [&](const F32 xPoint) {
        const F32 fraction = (xPoint + 1.0f) * 0.5f;
        const F32 y = points[1] + (points[3] - points[1]) * fraction;
        const auto value = Modules::detail::LineplotAmplitudeValue(y, config.rangeMin, config.rangeMax);
        REQUIRE(value.has_value());
        return jst::fmt::format("{:.1f} dBFS", *value);
    };

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Scroll, .position = {0.5f, 0.5f}, .scroll = {0, 90},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(interaction.zoom == Catch::Approx(10.0f));
    REQUIRE(interaction.offset == Catch::Approx(0.0f).margin(1e-6f));
    const auto padding = axis->paddingScale();

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursor.visible);
    REQUIRE(cursor.marker);
    REQUIRE(cursor.plot.x == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(cursor.plot.y == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(cursorText->get("cursor-x").fill == "100.0000 MHz");
    REQUIRE(cursorText->get("cursor-y").fill == "-50.0 dBFS");

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f + 0.25f * padding.x, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursor.visible);
    REQUIRE(cursor.marker);
    REQUIRE(std::abs(cursor.plot.x) <= padding.x);
    REQUIRE(cursor.plot.x == Catch::Approx(0.5f * padding.x));
    REQUIRE(cursor.plot.y == Catch::Approx(padding.y * 0.05f * std::tanh(1.0f)).margin(1e-5f));
    REQUIRE(cursorText->get("cursor-x").fill == "100.0500 MHz");
    REQUIRE(cursorText->get("cursor-y").fill == expectedAmplitude(0.05f));

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f + 0.49f * padding.x, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(cursor.visible);
    REQUIRE(std::abs(cursor.plot.x) <= padding.x);
    REQUIRE(cursor.plot.x == Catch::Approx(0.98f * padding.x));
    REQUIRE(cursorText->get("cursor-x").fill == "100.0980 MHz");
    REQUIRE(cursorText->get("cursor-y").fill == expectedAmplitude(0.098f));

    // Narrow surfaces can have negative horizontal padding while the center
    // still passes hit-testing. The cursor must stay centered after resizing.
    module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize,
                                         .size = {64, 512},
                                         .scale = 1.0f});
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f, 0.5f},
    });
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(axis->paddingScale().x == Catch::Approx(-0.25f));
    REQUIRE(cursor.visible);
    REQUIRE(cursor.marker);
    REQUIRE(cursor.plot.x == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(cursor.plot.y == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(cursorText->get("cursor-x").fill == "100.0000 MHz");
    REQUIRE(cursorText->get("cursor-y").fill == "-50.0 dBFS");

    REQUIRE(cursorText->destroy(&window) == Result::SUCCESS);
    REQUIRE(text->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Space holds displayed plots while processing continues and resumes at the latest data",
          "[modules][signal_view][present][keyboard][hold]") {
    const std::string mode = GENERATE("lineplot", "waterfall", "lineplot_waterfall");
    const U64 heldFrames = GENERATE(2, 11);
    const bool lineplot = Modules::detail::SignalViewHasLineplot(mode);
    const bool waterfall = Modules::detail::SignalViewHasWaterfall(mode);
    CAPTURE(mode, heldFrames);

    Tensor input(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(SetSignalAxes(input, {.sample = Index{1}, .batch = Index{0}}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].tensor = input;
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_view", DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic", module) == Result::SUCCESS);
    Modules::SignalView config;
    config.mode = mode;
    config.lineplotAveraging = 2;
    config.waterfallAveraging = 2;
    config.waterfallHeight = 4;
    config.maxHold = true;
    REQUIRE(module->create("plot", config, inputs) == Result::SUCCESS);
    Runtime runtime("plot", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"plot", module}}) == Result::SUCCESS);

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    Render::Components::Text::Config textConfig;
    textConfig.font = axisConfig.font;
    textConfig.maxCharacters = 256;
    textConfig.elements = {{"header", {}}, {"zoom", {}}, {"hold", {}},
                           {"amplitude-title", {}}, {"waterfall-title", {}}};
    auto text = std::make_shared<LabelTestText>(textConfig);
    REQUIRE(text->create(&window) == Result::SUCCESS);
    auto* impl = module->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    SignalViewImplAccess::wirePresentResources(*impl, axis, text);
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);

    std::vector<F32> displayedSignal(lineplot ? 16 : 0);
    std::vector<F32> displayedMaxHold(lineplot ? 16 : 0);
    std::vector<F32> displayedWaterfall(waterfall ? 32 : 0);
    std::vector<F32> uploadedWaterfall(waterfall ? (8 + 16) * 4 : 0);
    auto displayedUniforms = SignalViewImplAccess::waterfallOf(*impl).uniforms;
    const auto present = [&] {
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
        U64 bytes = ApplyBufferUploads(SignalViewImplAccess::lineplotOf(*impl).signalPointsBuffer,
                                       displayedSignal.data());
        bytes += ApplyBufferUploads(SignalViewImplAccess::lineplotOf(*impl).maxHoldPointsBuffer,
                                    displayedMaxHold.data());
        bytes += ApplyBufferUploads(SignalViewImplAccess::waterfallOf(*impl).binsBuffer,
                                    uploadedWaterfall.data());
        if (waterfall) {
            for (U64 row = 0; row < 4; ++row)
                std::copy_n(uploadedWaterfall.data() + row * (8 + 16), 8,
                            displayedWaterfall.data() + row * 8);
        }
        ApplyBufferUploads(SignalViewImplAccess::waterfallOf(*impl).uniformBuffer,
                            &displayedUniforms);
        return bytes;
    };
    std::unordered_set<std::string> skipped, failed;
    F32 average = -0.8f;
    const auto compute = [&](F32 value) {
        std::fill_n(input.data<F32>(), input.size(), value);
        REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
        REQUIRE(skipped.empty());
        REQUIRE(failed.empty());
        average = 0.5f * average + 0.5f * (2.0f * value - 1.0f);
        if (lineplot) {
            REQUIRE(ReadSignalPoints(module)[1] == Catch::Approx(std::tanh(2.0f * average)));
        }
    };
    const auto holdLabel = [&]() -> std::string {
        return lineplot ? text->get("hold").fill : "";
    };
    compute(0.1f);
    REQUIRE(present() > 0);
    REQUIRE(holdLabel() == (lineplot ? " " : ""));
    const auto frozenSignal = displayedSignal;
    const auto frozenMaxHold = displayedMaxHold;
    const auto frozenWaterfall = displayedWaterfall;
    const auto frozenIndex = displayedUniforms.index;

    compute(0.2f);
    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Space, {}});
    REQUIRE(present() == 0);
    REQUIRE(holdLabel() == (lineplot ? "HOLD" : ""));
    for (U64 frame = 0; frame < heldFrames; ++frame) {
        compute(0.3f + 0.05f * frame);
        module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Space, {}, true});
        REQUIRE(present() == 0);
        if (waterfall) {
            REQUIRE(ReadWaterfallHistory(module).writeIndex == (frame + 3) % config.waterfallHeight);
            REQUIRE(ReadWaterfallHistory(module).dirtyRows == std::min(frame + 2, config.waterfallHeight));
        }
    }
    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Release, KeyCode::Space, {}});
    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::A, {}});
    module->surface()->pushInputEvent(FocusEvent{false});
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Scroll, .position = {0.5f, 0.5f}, .scroll = {0, 1},
    });
    module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize, .size = {800, 600}});
    REQUIRE(present() == 0);
    REQUIRE(displayedSignal == frozenSignal);
    REQUIRE(displayedMaxHold == frozenMaxHold);
    REQUIRE(displayedWaterfall == frozenWaterfall);
    REQUIRE(displayedUniforms.index == frozenIndex);
    REQUIRE(holdLabel() == (lineplot ? "HOLD" : ""));
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).interaction.zoom > 1.0f);
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).interaction.viewSize.x == 800);
    if (waterfall) REQUIRE(displayedUniforms.zoom > 1.0f);

    module->surface()->pushInputEvent(FocusEvent{true});
    module->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Space, {}});
    REQUIRE(present() > 0);
    REQUIRE(holdLabel() == (lineplot ? " " : ""));
    if (lineplot) {
        REQUIRE(displayedSignal == ReadSignalPoints(module));
        REQUIRE(displayedMaxHold == ReadMaxHoldPoints(module));
        REQUIRE(displayedSignal != frozenSignal);
        REQUIRE(displayedMaxHold != frozenMaxHold);
    }
    if (waterfall) {
        REQUIRE(displayedWaterfall == ReadWaterfallBins(module));
        REQUIRE(displayedWaterfall != frozenWaterfall);
        REQUIRE(displayedUniforms.index == ReadWaterfallHistory(module).writeIndex / F32{4});
        REQUIRE(ReadWaterfallHistory(module).dirtyRows == 0);
    }
    REQUIRE(present() == 0);
    compute(0.95f);
    REQUIRE(present() > 0);
    if (lineplot) REQUIRE(displayedSignal == ReadSignalPoints(module));
    if (waterfall) REQUIRE(displayedWaterfall == ReadWaterfallBins(module));

    REQUIRE(text->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture, "Marker drags commit the release position with or without a preceding move",
                 "[modules][signal_view][markers][release][config-edits][regression]") {
    const bool moveBeforeRelease = GENERATE(false, true);
    CAPTURE(moveBeforeRelease);
    TestFlowgraph::SyntheticSourceBlockConfig source;
    source.bufferSize = 32;
    REQUIRE(flowgraph->blockCreate("source", source, {}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].requested("source", "signal");
    InteractiveSignalViewConfig config;
    config.markers = {0.0f};
    REQUIRE(flowgraph->blockCreate("plot", config, inputs) == Result::SUCCESS);
    const auto module = interactiveSignalView;
    auto* impl = module->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    const auto& positions = SignalViewImplAccess::canvasOf(*impl).markerPositions;
    const auto& interaction = SignalViewImplAccess::canvasOf(*impl).interaction;
    const auto& cursor = SignalViewImplAccess::canvasOf(*impl).cursor;

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.verticalScale = 0.5f;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    auto markerText = MakeMarkerText(axisConfig.font, window);
    SignalViewImplAccess::wirePresentResources(*impl, axis, {}, {}, markerText);
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);
    const auto present = [&] { REQUIRE(presenter->presentSubmit() == Result::SUCCESS); };
    const auto pending = [&] {
        return (impl->*SignalViewImplAccess::configChangePendingMember())();
    };

    module->surface()->pushSurfaceEvent({
        .type = SurfaceEventType::Resize,
        .size = {1000, 800},
    });
    present();
    REQUIRE(positions == std::vector<F32>{0.0f});
    const auto padding = axis->paddingScale();
    const Extent2D<F32> origin = {0.5f, 0.25f};

    // A release below the drag threshold must remain a click.
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Click, .button = MouseButton::Left, .position = origin,
    });
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Release, .button = MouseButton::Left,
        .position = {origin.x + 2.0f * interaction.scale / interaction.viewSize.x, origin.y},
    });
    present();
    REQUIRE(positions == std::vector<F32>{0.0f});
    REQUIRE_FALSE(pending());

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Click, .button = MouseButton::Left, .position = origin,
    });
    if (moveBeforeRelease) {
        present();
        module->surface()->pushInputEvent(MouseEvent{
            .type = MouseEventType::Move,
            .position = {origin.x + 0.125f * padding.x, origin.y},
        });
        present();
        REQUIRE(positions[0] == Catch::Approx(0.25f));
        REQUIRE_FALSE(pending());
    }
    const Extent2D<F32> release = {origin.x + 0.25f * padding.x, origin.y};
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Release, .button = MouseButton::Left, .position = release,
    });
    present();
    REQUIRE(positions[0] == Catch::Approx(0.5f));
    REQUIRE(cursor.position.x == Catch::Approx(release.x));
    REQUIRE(pending());
    REQUIRE_FALSE(interaction.dragging);
    REQUIRE(interaction.zoom == Catch::Approx(1.0f));
    REQUIRE(interaction.offset == Catch::Approx(0.0f));

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE_FALSE(pending());
    REQUIRE(interactiveSignalView == module);
    const auto stored = std::any_cast<std::vector<F32>>(viewBlock("plot").config.at("markers"));
    REQUIRE(stored.size() == 1);
    REQUIRE(stored[0] == Catch::Approx(0.5f));

    // Later motion must not continue a drag that has already been released.
    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = origin,
    });
    present();
    REQUIRE(positions[0] == Catch::Approx(0.5f));
    REQUIRE_FALSE(pending());

    REQUIRE(markerText->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture, "Signal View markers persist through the owning block and follow external edits",
                 "[modules][signal_view][markers][config-edits]") {
    TestFlowgraph::SyntheticSourceBlockConfig source;
    source.bufferSize = 32;
    REQUIRE(flowgraph->blockCreate("source", source, {}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].requested("source", "signal");
    REQUIRE(flowgraph->blockCreate("plot", InteractiveSignalViewConfig{}, inputs) == Result::SUCCESS);
    const auto original = interactiveSignalView;
    auto* impl = original->getImpl<Modules::SignalViewImpl>();
    REQUIRE(impl);
    const auto& cursor = SignalViewImplAccess::canvasOf(*impl).cursor;

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.verticalScale = 0.5f;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    Render::Components::Text::Config textConfig;
    textConfig.font = axisConfig.font;
    textConfig.maxCharacters = 64;
    textConfig.elements = {{"header", {}}, {"zoom", {}}, {"hold", {}},
                           {"amplitude-title", {}}, {"waterfall-title", {}}};
    auto text = std::make_shared<LabelTestText>(textConfig);
    REQUIRE(text->create(&window) == Result::SUCCESS);
    auto markerText = MakeMarkerText(axisConfig.font, window);
    SignalViewImplAccess::wirePresentResources(*impl, axis, text, {}, markerText);
    auto* presenter = original->getImpl<Scheduler::Context>();
    REQUIRE(presenter);
    const auto present = [&] {
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    };
    const auto pending = [&] {
        return (impl->*SignalViewImplAccess::configChangePendingMember())();
    };
    const auto storedMarkers = [&] {
        return std::any_cast<std::vector<F32>>(viewBlock("plot").config.at("markers"));
    };
    const auto storedPins = [&] {
        return std::any_cast<std::vector<U64>>(viewBlock("plot").config.at("pins"));
    };
    const auto spanLabel = [&](const U64 index) {
        return markerText->get(jst::fmt::format("span-{}-label", index)).fill;
    };

    original->surface()->pushSurfaceEvent({
        .type = SurfaceEventType::Resize,
        .size = {1000, 800},
    });
    present();
    REQUIRE(storedMarkers().empty());

    original->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f, 0.25f},
    });
    present();
    REQUIRE(cursor.visible);
    original->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Click,
        .button = MouseButton::Left,
        .position = {0.5f, 0.25f},
        .modifiers = {.shift = true},
    });
    present();
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions == std::vector<F32>{0.0f});
    REQUIRE(pending());
    REQUIRE(storedMarkers().empty());
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE_FALSE(pending());
    REQUIRE(interactiveSignalView == original);
    REQUIRE(storedMarkers() == std::vector<F32>{0.0f});
    REQUIRE(impl->markers == std::vector<F32>{0.0f});
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions == std::vector<F32>{0.0f});
    REQUIRE(markerText->get("marker-0-tag").fill == "M1");

    Parser::Map edit;
    edit["markers"] = std::vector<F32>{-0.5f, 0.25f};
    REQUIRE(flowgraph->blockReconfigure("plot", edit) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE(interactiveSignalView == original);
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions == std::vector<F32>{-0.5f, 0.25f});
    REQUIRE(markerText->get("marker-1-tag").fill == "M2");
    REQUIRE(storedPins().empty());
    REQUIRE(spanLabel(0) == " ");

    const auto tag = markerText->get("marker-0-tag").position;
    const Extent2D<F32> tagPosition = {(tag.x + 1.0f) * 0.5f, (1.0f - tag.y) * 0.5f};
    original->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Click, .button = MouseButton::Left, .position = tagPosition,
    });
    original->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Release, .button = MouseButton::Left, .position = tagPosition,
    });
    original->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Move, .position = {0.5f, 0.75f},
    });
    present();
    REQUIRE(pending());
    REQUIRE(spanLabel(0) == "0.7500");
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE_FALSE(pending());
    REQUIRE(storedPins() == std::vector<U64>{0});
    REQUIRE(impl->pins == std::vector<U64>{0});
    REQUIRE(spanLabel(0) == "0.7500");

    edit.clear();
    edit["pins"] = std::vector<U64>{};
    REQUIRE(flowgraph->blockReconfigure("plot", edit) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE(spanLabel(0) == " ");

    edit["pins"] = std::vector<U64>{1};
    REQUIRE(flowgraph->blockReconfigure("plot", edit) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE(spanLabel(0) == "0.7500");

    original->surface()->pushInputEvent(KeyEvent{KeyEventType::Press, KeyCode::Escape, {}});
    present();
    REQUIRE(pending());
    REQUIRE(spanLabel(0) == " ");
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE_FALSE(pending());
    REQUIRE(storedPins().empty());

    edit["pins"] = std::vector<U64>{1};
    REQUIRE(flowgraph->blockReconfigure("plot", edit) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE(spanLabel(0) == "0.7500");

    original->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Click,
        .button = MouseButton::Right,
        .position = {0.5f, 0.25f},
        .modifiers = {.shift = true},
    });
    present();
    REQUIRE(SignalViewImplAccess::canvasOf(*impl).markerPositions.empty());
    REQUIRE(pending());
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE_FALSE(pending());
    REQUIRE(storedMarkers().empty());
    REQUIRE(storedPins().empty());
    REQUIRE(markerText->get("marker-0-tag").fill == " ");

    REQUIRE(markerText->destroy(&window) == Result::SUCCESS);
    REQUIRE(text->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
}

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
    auto& splitter = SignalViewImplAccess::canvasOf(*impl).splitter;
    auto& interaction = SignalViewImplAccess::canvasOf(*impl).interaction;

    LabelTestWindow window;
    Render::Components::Axis::Config axisConfig;
    axisConfig.verticalScale = 0.5f;
    axisConfig.font = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    auto axis = std::make_shared<LabelTestAxis>(axisConfig);
    REQUIRE(axis->create(&window) == Result::SUCCESS);
    Render::Components::Text::Config textConfig;
    textConfig.font = axisConfig.font;
    textConfig.maxCharacters = 64;
    textConfig.elements = {{"header", {}}, {"zoom", {}}, {"hold", {}},
                           {"amplitude-title", {}}, {"waterfall-title", {}}};
    auto text = std::make_shared<LabelTestText>(textConfig);
    REQUIRE(text->create(&window) == Result::SUCCESS);
    SignalViewImplAccess::wirePresentResources(*impl, axis, text);
    auto* presenter = original->getImpl<Scheduler::Context>();
    REQUIRE(presenter);
    const auto present = [&] {
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    };
    const auto pending = [&] {
        return (impl->*SignalViewImplAccess::configChangePendingMember())();
    };
    original->surface()->pushSurfaceEvent({
        .type = SurfaceEventType::Resize,
        .size = {1000, 800},
    });
    present();
    const auto panels = Modules::detail::CalculateSignalViewPanels(
        axis->paddingScale(), interaction.viewSize, splitter.ratio);
    const auto positionAtRatio = [&](F32 ratio) {
        return (panels.plot.y + panels.plot.height * ratio) / interaction.viewSize.y;
    };
    MouseEvent event{};
    event.type = MouseEventType::Click;
    event.button = MouseButton::Left;
    event.position = {0.5f, positionAtRatio(0.5f)};
    original->surface()->pushInputEvent(event);
    present();
    REQUIRE(splitter.dragging);
    REQUIRE_FALSE(interaction.dragging);

    event.type = MouseEventType::Move;
    event.position = {0.7f, positionAtRatio(0.7f)};
    original->surface()->pushInputEvent(event);
    present();
    REQUIRE(splitter.ratio == Catch::Approx(0.7f));
    REQUIRE(text->get("zoom").fill == "SPLIT 70%");
    REQUIRE(axis->getConfig().verticalScale ==
            Catch::Approx(0.7f).margin(1.0f / panels.plot.height));
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
    original->surface()->pushInputEvent(event);
    present();
    REQUIRE_FALSE(splitter.dragging);
    REQUIRE(pending());
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    present();
    REQUIRE_FALSE(pending());
    REQUIRE(interactiveSignalView == original);
    REQUIRE(std::any_cast<F32>(viewBlock("plot").config.at("splitRatio")) == Catch::Approx(0.7f));
    REQUIRE(impl->splitRatio == splitter.ratio);

    const auto committedRatio = splitter.ratio;
    event.type = MouseEventType::Move;
    event.position = {0.2f, 0.2f};
    original->surface()->pushInputEvent(event);
    present();
    REQUIRE_FALSE(splitter.dragging);
    REQUIRE_FALSE(pending());
    REQUIRE(splitter.ratio == committedRatio);

    Parser::Map sidebar;
    sidebar["splitRatio"] = F32{0.4f};
    REQUIRE(flowgraph->blockReconfigure("plot", sidebar) == Result::SUCCESS);
    present();
    REQUIRE(splitter.ratio == 0.4f);
    REQUIRE(axis->getConfig().verticalScale ==
            Catch::Approx(0.4f).margin(1.0f / panels.plot.height));
    REQUIRE_FALSE(pending());

    REQUIRE(text->destroy(&window) == Result::SUCCESS);
    REQUIRE(axis->destroy(&window) == Result::SUCCESS);
}
#endif

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
            config.lineplotAveraging = 2;
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

            Parser::Map markerEdit;
            markerEdit["markers"] = std::vector<F32>{-0.5f, 0.25f};
            REQUIRE(module->reconfigure(markerEdit) == Result::SUCCESS);
            REQUIRE(static_cast<const Modules::SignalView&>(module->config()).markers ==
                    std::vector<F32>{-0.5f, 0.25f});
            REQUIRE((impl->*SignalViewImplAccess::signalPointsMember()).id() == signalId);
            REQUIRE(ReadSignalPoints(module) == trace);
            for (const auto& invalid : {std::vector<F32>{2.0f},
                                        std::vector<F32>{std::numeric_limits<F32>::quiet_NaN()},
                                        std::vector<F32>(Modules::detail::MaxMarkers + 1, 0.0f)}) {
                markerEdit["markers"] = invalid;
                REQUIRE(module->reconfigure(markerEdit) == Result::ERROR);
                REQUIRE(static_cast<const Modules::SignalView&>(module->config()).markers ==
                        std::vector<F32>{-0.5f, 0.25f});
            }
            markerEdit["markers"] = std::vector<F32>{-0.5f, 0.25f};
            REQUIRE(module->reconfigure(markerEdit) == Result::SUCCESS);

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

    const std::array<const char*, 4> modes = {
        "lineplot",
        "waterfall",
        "lineplot_waterfall",
        "waterfall_3d",
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
                config.lineplotAveraging = 4;
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
                config.lineplotAveraging = 0;
                ctx.setConfig(config);
                ctx.setInput("signal", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("waterfall averaging") {
                Modules::SignalView config;
                config.mode = "waterfall";
                config.waterfallAveraging = 0;
                RequireSignalViewValidationError(implementation, config, input);
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

            RequireSignalViewValidationError(implementation,
                                             Modules::SignalView{},
                                             DataType::F32,
                                             {1});
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

TEST_CASE("Combined signal view keeps full-resolution traces and waterfall rows",
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
            REQUIRE(ReadSignalPoints(module).size() == rowWidth * 2);
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
                    RequirePoints(leadingSnapshot.signalPoints, {
                        -1.0f, -0.76159416f, 0.0f, 0.46211716f, 1.0f, 0.46211716f,
                    });
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

TEST_CASE("Signal View configuration omits fixed and removed rendering settings",
          "[modules][signal_view][config]") {
    Modules::SignalView config;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(serialized.contains("interpolate"));
    REQUIRE_FALSE(serialized.contains("waterfallInterpolate"));
    REQUIRE_FALSE(serialized.contains("thickness"));
    REQUIRE_FALSE(serialized.contains("decimation"));
}

TEST_CASE("Signal View serializes plot labels", "[modules][signal_view][config]") {
    Modules::SignalView config;
    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(serialized.at("lineplotAveraging")) == 1);
    REQUIRE(std::any_cast<U64>(serialized.at("waterfallAveraging")) == 1);
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

TEST_CASE("Waterfall emits complete averaged rows across batches and runtime rebuilds",
          "[modules][signal_view][waterfall][averaging][layout]") {
    const U64 batches = GENERATE(U64{1}, U64{2}, U64{6}, U64{18});
    const U64 width = GENERATE(U64{1}, U64{3});
    const bool trailingBatch = GENERATE(false, true);
    const U64 averaging = GENERATE(U64{1}, U64{2}, U64{4}, U64{7});
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE_FALSE(implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device) {
            CAPTURE(batches, width, trailingBatch, averaging);
            constexpr U64 height = 5;
            const auto sample = [](const U64 row, const U64 element) -> F32 {
                return static_cast<F32>((row * 3 + element * 5 + 1) % 8) / 8.0f;
            };
            const auto expected = [&](const U64 row, const U64 element) -> F32 {
                F64 sum = 0.0;
                for (U64 offset = 0; offset < averaging; ++offset) {
                    sum += sample(row * averaging + offset, element);
                }
                return static_cast<F32>(sum / static_cast<F64>(averaging));
            };

            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32,
                                  trailingBatch ? Shape{width, batches} : Shape{batches, width},
                                  {.hostAccessible = true}) == Result::SUCCESS);
            const Index sampleAxis = trailingBatch ? 0 : 1;
            const Index batchAxis = trailingBatch ? 1 : 0;
            REQUIRE(SetSignalAxes(input, {.sample = sampleAxis, .batch = batchAxis}) ==
                    Result::SUCCESS);
            TensorMap inputs;
            inputs["signal"].tensor = input;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view", implementation.device,
                                          implementation.runtime, implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallAveraging = averaging;
            config.waterfallHeight = height;
            REQUIRE(module->create("waterfall", config, inputs) == Result::SUCCESS);
            Runtime runtime("waterfall", implementation.device, implementation.runtime);
            REQUIRE(runtime.create({{"waterfall", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped, failed;

            for (U64 start = 0; start < 36; start += batches) {
                for (U64 row = 0; row < batches; ++row) {
                    for (U64 element = 0; element < width; ++element) {
                        input.data<F32>()[row * input.stride(batchAxis) +
                                          element * input.stride(sampleAxis)] = sample(start + row, element);
                    }
                }
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
                const U64 total = (start + batches) / averaging;
                const U64 first = total > height ? total - height : 0;
                const auto bins = ReadWaterfallBins(module);
                for (U64 row = first; row < total; ++row) {
                    for (U64 element = 0; element < width; ++element) {
                        REQUIRE(bins[(row % height) * width + element] ==
                                Catch::Approx(expected(row, element)).margin(1e-6));
                    }
                }
                for (U64 row = total; row < height; ++row) {
                    for (U64 element = 0; element < width; ++element) {
                        REQUIRE(bins[row * width + element] == 0.0f);
                    }
                }
                REQUIRE(ReadWaterfallHistory(module).writeIndex == total % height);
                REQUIRE(ReadWaterfallHistory(module).dirtyRows == std::min(total, height));
                if (start + batches == 18) {
                    REQUIRE(runtime.destroy() == Result::SUCCESS);
                    REQUIRE(runtime.create({{"waterfall", module}}) == Result::SUCCESS);
                }
            }
            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Waterfall averaging counts complete groups without overflowing",
          "[modules][signal_view][waterfall][averaging]") {
    constexpr U64 limit = std::numeric_limits<U64>::max();
    const auto complete = Modules::PlanWaterfallAveraging(limit - 1, limit, limit);
    REQUIRE(complete.rowCount == 1);
    REQUIRE(complete.pendingRows == limit - 1);
    const auto partial = Modules::PlanWaterfallAveraging(limit - 2, 1, limit);
    REQUIRE(partial.rowCount == 0);
    REQUIRE(partial.pendingRows == limit - 1);
    const auto raw = Modules::PlanWaterfallAveraging(0, limit, 1);
    REQUIRE(raw.rowCount == limit);
    REQUIRE(raw.pendingRows == 0);
}

TEST_CASE("Waterfall group averages retain finite extreme amplitudes",
          "[modules][signal_view][waterfall][averaging][numeric]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE_FALSE(implementations.empty());
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {4, 2},
                                  {.hostAccessible = true}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {.sample = Index{1}, .batch = Index{0}}) == Result::SUCCESS);
            const F32 large = std::numeric_limits<F32>::max();
            const F32 small = std::numeric_limits<F32>::min();
            for (U64 row = 0; row < 4; ++row) {
                input.at<F32>(row, 0) = large;
                input.at<F32>(row, 1) = small;
            }
            TensorMap inputs;
            inputs["signal"].tensor = input;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view", implementation.device,
                                          implementation.runtime, implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "waterfall";
            config.waterfallAveraging = 4;
            config.waterfallHeight = 2;
            REQUIRE(module->create("waterfall", config, inputs) == Result::SUCCESS);
            Runtime runtime("waterfall", implementation.device, implementation.runtime);
            REQUIRE(runtime.create({{"waterfall", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped, failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            REQUIRE(ReadWaterfallBins(module) == std::vector<F32>{large, small, 0.0f, 0.0f});
            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View applies independent lineplot and waterfall averaging strengths",
          "[modules][signal_view][averaging]") {
    const U64 lineplotAveraging = GENERATE(U64{1}, U64{2}, U64{4});
    const U64 waterfallAveraging = GENERATE(U64{1}, U64{2}, U64{4});
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE_FALSE(implementations.empty());
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device) {
            CAPTURE(lineplotAveraging, waterfallAveraging);
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2},
                                  {.hostAccessible = true}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) == Result::SUCCESS);
            input.at<F32>(0) = 0.25f;
            input.at<F32>(1) = 0.75f;
            TensorMap inputs;
            inputs["signal"].tensor = input;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view", implementation.device,
                                          implementation.runtime, implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.lineplotAveraging = lineplotAveraging;
            config.waterfallAveraging = waterfallAveraging;
            config.maxHold = true;
            config.waterfallHeight = 2;
            REQUIRE(module->create("plot", config, inputs) == Result::SUCCESS);
            Runtime runtime("plot", implementation.device, implementation.runtime);
            REQUIRE(runtime.create({{"plot", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped, failed;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);

            input.at<F32>(0) = 0.75f;
            input.at<F32>(1) = 0.25f;
            REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
            const F32 lineValue = std::tanh(-1.0f + 2.0f / static_cast<F32>(lineplotAveraging));
            RequirePoints(ReadSignalPoints(module), {-1.0f, lineValue, 1.0f, -lineValue});
            const std::vector<F32> expectedWaterfall = waterfallAveraging == 1
                ? std::vector<F32>{0.25f, 0.75f, 0.75f, 0.25f}
                : (waterfallAveraging == 2
                    ? std::vector<F32>{0.5f, 0.5f, 0.0f, 0.0f}
                    : std::vector<F32>(4, 0.0f));
            REQUIRE(ReadWaterfallBins(module) == expectedWaterfall);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == (2 / waterfallAveraging) % 2);
            REQUIRE(ReadWaterfallHistory(module).dirtyRows == 2 / waterfallAveraging);
            const F32 firstHold = lineplotAveraging > 2 ? -1.0f : lineValue;
            const F32 secondHold = lineplotAveraging == 1 ? lineValue
                : (lineplotAveraging == 2 ? -lineValue : -1.0f);
            RequirePoints(ReadMaxHoldPoints(module), {
                -1.0f, firstHold, 1.0f, secondHold,
            });
            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Waterfall averaging edits preserve history and discard affected partial groups",
          "[modules][signal_view][waterfall][averaging][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE_FALSE(implementations.empty());
    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device) {
            Tensor input;
            REQUIRE(input.create(implementation.device, DataType::F32, {2},
                                  {.hostAccessible = true}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) == Result::SUCCESS);
            TensorMap inputs;
            inputs["signal"].tensor = input;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("signal_view", implementation.device,
                                          implementation.runtime, implementation.provider,
                                          module) == Result::SUCCESS);
            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.lineplotAveraging = 8;
            config.waterfallAveraging = 2;
            config.maxHold = true;
            config.waterfallHeight = 16;
            REQUIRE(module->create("plot", config, inputs) == Result::SUCCESS);
            Runtime runtime("plot", implementation.device, implementation.runtime);
            REQUIRE(runtime.create({{"plot", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skipped, failed;
            const auto compute = [&](const F32 value, const std::optional<F32> expected = std::nullopt) {
                CAPTURE(value, expected);
                const auto previousHistory = ReadWaterfallHistory(module);
                const auto previousBins = ReadWaterfallBins(module);
                std::fill_n(input.data<F32>(), input.size(), value);
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
                const auto history = ReadWaterfallHistory(module);
                const auto bins = ReadWaterfallBins(module);
                if (expected) {
                    REQUIRE(history.writeIndex == (previousHistory.writeIndex + 1) % config.waterfallHeight);
                    REQUIRE(bins[previousHistory.writeIndex * 2] == *expected);
                    REQUIRE(bins[previousHistory.writeIndex * 2 + 1] == *expected);
                } else {
                    REQUIRE(history.writeIndex == previousHistory.writeIndex);
                    REQUIRE(history.dirtyRows == previousHistory.dirtyRows);
                    REQUIRE(bins == previousBins);
                }
            };
            compute(0.25f);
            Parser::Map strength;
            strength["waterfallAveraging"] = U64{4};
            REQUIRE(module->reconfigure(strength, true) == Result::SUCCESS);
            compute(0.75f, 0.5f);
            compute(0.75f);
            const auto history = ReadWaterfallHistory(module);
            const auto beforeEdit = ReadWaterfallBins(module);
            const auto hold = ReadMaxHoldPoints(module);
            const auto* impl = module->getImpl<Modules::SignalViewImpl>();
            const auto warmup = impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember();
            REQUIRE(module->reconfigure(strength) == Result::SUCCESS);
            REQUIRE(ReadWaterfallBins(module) == beforeEdit);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == history.writeIndex);
            REQUIRE(ReadMaxHoldPoints(module) == hold);
            REQUIRE(impl->*SignalViewImplAccess::maxHoldWarmupBlocksMember() == warmup);
            compute(0.0f);
            compute(1.0f);

            Parser::Map invalid;
            invalid["waterfallAveraging"] = U64{0};
            REQUIRE(module->reconfigure(invalid) == Result::ERROR);
            compute(1.0f);

            REQUIRE(module->reconfigure(strength) == Result::SUCCESS);
            Parser::Map lineplot;
            lineplot["lineplotAveraging"] = U64{4};
            REQUIRE(module->reconfigure(lineplot) == Result::SUCCESS);
            compute(0.0f, 0.5f);
            compute(0.75f);
            lineplot["lineplotAveraging"] = U64{1};
            REQUIRE(module->reconfigure(lineplot) == Result::SUCCESS);
            compute(0.25f);
            compute(0.5f);
            compute(0.5f, 0.5f);
            compute(0.25f);

            strength["waterfallAveraging"] = U64{1};
            REQUIRE(module->reconfigure(strength) == Result::SUCCESS);
            compute(0.75f, 0.75f);
            strength["waterfallAveraging"] = U64{4};
            REQUIRE(module->reconfigure(strength) == Result::SUCCESS);
            compute(0.5f);
            compute(1.0f);

            Parser::Map range;
            range["rangeMin"] = F32{-50.0f};
            REQUIRE(module->reconfigure(range) == Result::SUCCESS);
            REQUIRE(ReadWaterfallHistory(module).writeIndex == 0);
            REQUIRE(ReadWaterfallBins(module) == std::vector<F32>(32, 0.0f));
            compute(0.25f);
            compute(0.25f);
            compute(0.25f);
            compute(0.25f, 0.25f);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Lineplot helpers initialize points and preserve max-hold warmup",
           "[modules][signal_view][lineplot][regression]") {
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

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal View displays externally subsampled data after ratio changes",
                 "[modules][signal_view][decimator][integration]") {
    TestFlowgraph::SyntheticSourceBlockConfig source;
    source.bufferSize = 8;
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
    Tensor input = viewBlock("src").outputs.at("signal").tensor;
    REQUIRE(SetSignalAxes(input, {.sample = Index{0}}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleRate", F32{48000.0f}) == Result::SUCCESS);
    for (U64 index = 0; index < input.size(); ++index) {
        input.at<F32>(index) = static_cast<F32>(index) / 8.0f;
    }

    Blocks::Decimator decimator;
    decimator.method = "subsample";
    decimator.ratio = 2;
    TensorMap decimatorInputs;
    decimatorInputs["buffer"].requested("src", "signal");
    REQUIRE(flowgraph->blockCreate("decimator", decimator, decimatorInputs) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("decimator", "buffer");
    REQUIRE(flowgraph->blockCreate("view", InteractiveSignalViewConfig{}, inputs) ==
            Result::SUCCESS);

    for (const U64 ratio : {U64{2}, U64{4}}) {
        Parser::Map update;
        update["ratio"] = ratio;
        REQUIRE(flowgraph->blockReconfigure("decimator", update) == Result::SUCCESS);
        REQUIRE(viewBlock("view").state == Block::State::Created);
        REQUIRE(flowgraph->compute() == Result::SUCCESS);

        const Tensor received = viewBlock("view").inputs.at("signal").tensor;
        const U64 width = input.size() / ratio;
        REQUIRE(received.shape() == Shape{width});
        REQUIRE(std::any_cast<F32>(received.attribute("sampleRate")) ==
                48000.0f / static_cast<F32>(ratio));
        const auto points = ReadSignalPoints(interactiveSignalView);
        const auto bins = ReadWaterfallBins(interactiveSignalView);
        REQUIRE(points.size() == width * 2);
        REQUIRE(bins.size() == width * 8);
        for (U64 index = 0; index < width; ++index) {
            const F32 expected = input.at<F32>(index * ratio);
            REQUIRE(points[index * 2] == Catch::Approx(
                static_cast<F32>(index) * 2.0f / (width - 1) - 1.0f));
            REQUIRE(points[index * 2 + 1] == Catch::Approx(std::tanh(expected * 4.0f - 2.0f)));
            REQUIRE(bins[index] == expected);
        }
    }
    interactiveSignalView.reset();
}

TEST_CASE("Amplitude and Range feed true log averages to both Signal View traces",
          "[modules][signal_view][averaging][integration][numeric]") {
    const bool trailingBatch = GENERATE(false, true);
    const bool narrowRange = GENERATE(false, true);
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE_FALSE(implementations.empty());
    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            CAPTURE(trailingBatch, narrowRange);
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::CF32, {2, 2},
                                  {.hostAccessible = true}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = trailingBatch ? Index{0} : Index{1},
                .batch = trailingBatch ? Index{1} : Index{0},
            }) == Result::SUCCESS);

            const auto build = [&](const std::string& type, const Module::Config& config,
                                    const TensorLink& signal) {
                std::shared_ptr<Module> module;
                REQUIRE(Registry::BuildModule(type, impl.device, impl.runtime,
                                              impl.provider, module) == Result::SUCCESS);
                REQUIRE(module->create(type, config, {{"signal", signal}}) == Result::SUCCESS);
                return module;
            };
            TensorMap inputs;
            inputs["signal"].tensor = input;
            auto amplitude = build("amplitude", Modules::Amplitude{}, inputs.at("signal"));
            Modules::Range rangeConfig;
            rangeConfig.min = narrowRange ? -80.0f : -100.0f;
            rangeConfig.max = narrowRange ? -40.0f : 0.0f;
            auto range = build("range", rangeConfig, amplitude->outputs().at("signal"));
            Modules::SignalView config;
            config.mode = "lineplot_waterfall";
            config.lineplotAveraging = 2;
            config.waterfallAveraging = 2;
            config.waterfallHeight = 4;
            config.maxHold = true;
            config.rangeMin = rangeConfig.min;
            config.rangeMax = rangeConfig.max;
            auto plot = build("signal_view", config, range->outputs().at("signal"));
            const Runtime::Modules modules = {
                {"amplitude", amplitude}, {"range", range}, {"signal_view", plot},
            };
            Runtime runtime("log_averaging", impl.device, impl.runtime);
            REQUIRE(runtime.create(modules) == Result::SUCCESS);

            const auto compute = [&](const F32 firstDb, const F32 secondDb,
                                      const F32 rowDb, const F32 traceDb) {
                CAPTURE(firstDb, secondDb, rowDb, traceDb);
                const F32 levels[] = {firstDb, secondDb};
                for (U64 batch = 0; batch < 2; ++batch) {
                    const CF32 value{2.0f * std::pow(10.0f, levels[batch] / 20.0f), 0.0f};
                    for (U64 bin = 0; bin < 2; ++bin) {
                        input.at<CF32>(trailingBatch ? bin : batch,
                                       trailingBatch ? batch : bin) = value;
                    }
                }
                const U64 row = ReadWaterfallHistory(plot).writeIndex;
                std::unordered_set<std::string> skipped, failed;
                REQUIRE(runtime.compute({"amplitude", "range", "signal_view"},
                                         skipped, failed) == Result::SUCCESS);
                REQUIRE(skipped.empty());
                REQUIRE(failed.empty());
                const auto points = ReadSignalPoints(plot);
                const auto bins = ReadWaterfallBins(plot);
                const F32 span = config.rangeMax - config.rangeMin;
                const F32 displayed = std::tanh(4.0f * (traceDb - config.rangeMin) / span - 2.0f);
                for (U64 bin = 0; bin < 2; ++bin) {
                    REQUIRE(points[bin * 2 + 1] == Catch::Approx(displayed).margin(0.001f));
                    const F32 measuredDb = config.rangeMin + bins[row * 2 + bin] * span;
                    REQUIRE(measuredDb == Catch::Approx(rowDb).margin(0.02f));
                }
                for (const F32 point : ReadMaxHoldPoints(plot)) {
                    REQUIRE(std::isfinite(point));
                    REQUIRE(point >= -1.0f);
                    REQUIRE(point <= 1.0f);
                }
            };

            SECTION("unclipped averages, runtime rebuilds, and reseeding") {
                compute(-100.0f, -50.0f, -75.0f, -75.0f);
                compute(100.0f, 100.0f, 100.0f, 12.5f);
                compute(-100.0f, -100.0f, -100.0f, -43.75f);
                REQUIRE(runtime.destroy() == Result::SUCCESS);
                REQUIRE(runtime.create(modules) == Result::SUCCESS);
                compute(-25.0f, -25.0f, -25.0f, -34.375f);

                config.rangeMin = -200.0f;
                config.rangeMax = 0.0f;
                REQUIRE(range->reconfigure({{"min", config.rangeMin},
                                             {"max", config.rangeMax}}) == Result::SUCCESS);
                REQUIRE(plot->reconfigure({{"rangeMin", config.rangeMin},
                                            {"rangeMax", config.rangeMax}}) == Result::SUCCESS);
                compute(-100.0f, -50.0f, -75.0f, -75.0f);
                REQUIRE(plot->reconfigure({{"lineplotAveraging", U64{4}}}) == Result::SUCCESS);
                compute(-25.0f, -25.0f, -25.0f, -25.0f);
                compute(-100.0f, -50.0f, -75.0f, -37.5f);
            }
            SECTION("zero bins use the display endpoint at high averaging strength") {
                constexpr U64 strength = 256;
                REQUIRE(plot->reconfigure({{"lineplotAveraging", strength}}) == Result::SUCCESS);
                const F32 silence = -std::numeric_limits<F32>::infinity();
                F32 traceDb = config.rangeMin;
                compute(silence, silence, traceDb, traceDb);
                for (U64 step = 0; step < 6; ++step) {
                    traceDb += (-50.0f - traceDb) / static_cast<F32>(strength);
                    compute(-50.0f, -50.0f, -50.0f, traceDb);
                }
            }
            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(plot->destroy() == Result::SUCCESS);
            REQUIRE(range->destroy() == Result::SUCCESS);
            REQUIRE(amplitude->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View averages keep updating after NaN and infinity inputs",
          "[modules][signal_view][lineplot][waterfall][averaging][regression]") {
    const U64 lineplotAveraging = GENERATE(U64{1}, U64{4});
    const U64 waterfallAveraging = GENERATE(U64{1}, U64{4});
    const bool existingHistory = GENERATE(false, true);
    const auto implementations = Registry::ListAvailableModules("signal_view");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            CAPTURE(lineplotAveraging, waterfallAveraging, existingHistory);
            Tensor cpuInput(DeviceType::CPU, DataType::F32, {2, 4});
            REQUIRE(SetSignalAxes(cpuInput, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);
            std::fill_n(cpuInput.data<F32>(), cpuInput.size(), 0.25f);

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
            config.lineplotAveraging = lineplotAveraging;
            config.waterfallAveraging = waterfallAveraging;
            config.waterfallHeight = 2;
            config.maxHold = true;
            config.fill = false;
            REQUIRE(module->create("signal_view", config, inputs) == Result::SUCCESS);

            Runtime runtime("signal_view", implementation.device,
                            implementation.runtime);
            REQUIRE(runtime.create({{"signal_view", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skipped;
            std::unordered_set<std::string> failed;
            const auto compute = [&] {
                if (implementation.device != DeviceType::CPU) {
                    REQUIRE(input.copyFrom(cpuInput) == Result::SUCCESS);
                }
                REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
                REQUIRE(skipped.empty());
                REQUIRE(failed.empty());
                for (const F32 value : ReadWaterfallBins(module)) {
                    REQUIRE(std::isfinite(value));
                }
                for (const F32 value : ReadMaxHoldPoints(module)) {
                    REQUIRE(std::isfinite(value));
                    REQUIRE(value >= -1.0f);
                    REQUIRE(value <= 1.0f);
                }
            };
            if (existingHistory) {
                compute();
            }

            const F32 infinity = std::numeric_limits<F32>::infinity();
            cpuInput.at<F32>(0, 0) = std::numeric_limits<F32>::quiet_NaN();
            cpuInput.at<F32>(0, 1) = infinity;
            cpuInput.at<F32>(0, 2) = -infinity;
            cpuInput.at<F32>(0, 3) = infinity;
            for (U64 bin = 0; bin < 3; ++bin) {
                cpuInput.at<F32>(1, bin) = 0.75f;
            }
            cpuInput.at<F32>(1, 3) = -infinity;
            compute();

            std::array<F32, 4> expected = {-0.25f, 0.75f, -0.25f, 0.0f};
            if (existingHistory) {
                for (auto& value : expected) {
                    value = -0.5f + (value + 0.5f) / static_cast<F32>(lineplotAveraging);
                }
            }
            const auto firstPoints = ReadSignalPoints(module);
            for (U64 bin = 0; bin < 4; ++bin) {
                REQUIRE(firstPoints[bin * 2 + 1] == Catch::Approx(std::tanh(2.0f * expected[bin])).margin(1e-6f));
            }
            if (waterfallAveraging == 1) {
                REQUIRE(ReadWaterfallBins(module) == std::vector<F32>{
                    0.0f, 1.0f, 0.0f, 1.0f, 0.75f, 0.75f, 0.75f, 0.0f,
                });
            } else if (existingHistory) {
                REQUIRE(ReadWaterfallBins(module) == std::vector<F32>{
                    0.3125f, 0.5625f, 0.3125f, 0.375f, 0.0f, 0.0f, 0.0f, 0.0f,
                });
            }

            std::fill_n(cpuInput.data<F32>(), cpuInput.size(), 0.75f);
            for (U64 step = 1; step <= 8; ++step) {
                compute();
                const auto points = ReadSignalPoints(module);
                const F32 decay = std::pow(1.0f - 1.0f / static_cast<F32>(lineplotAveraging),
                                           static_cast<F32>(step));
                for (U64 bin = 0; bin < 4; ++bin) {
                    const F32 recovered = 0.5f + (expected[bin] - 0.5f) * decay;
                    REQUIRE(points[bin * 2 + 1] == Catch::Approx(std::tanh(2.0f * recovered)).margin(1e-6f));
                }
            }
            REQUIRE(ReadWaterfallBins(module) == std::vector<F32>(8, 0.75f));

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
            config.lineplotAveraging = 4;
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
                REQUIRE(freshPoints[(index * 2) + 1] == Catch::Approx(-0.76159416f));
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
            config.lineplotAveraging = 4;
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
            averaging["lineplotAveraging"] = U64{16};
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
                REQUIRE(freshPoints[(index * 2) + 1] == Catch::Approx(-0.76159416f));
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
            config.lineplotAveraging = 4;
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
            averaging["lineplotAveraging"] = U64{16};
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
            config.lineplotAveraging = 1;
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
                REQUIRE(hold[(index * 2) + 1] == Catch::Approx(-0.76159416f));
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View max hold captures the seeded trace after its configured warmup",
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
            config.lineplotAveraging = 4;
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
                REQUIRE(hold[(index * 2) + 1] == Catch::Approx(-0.76159416f));
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal View surface sweep order visits every cell far to near",
          "[modules][signal_view][surface][scene]") {
    const I32 count = GENERATE(I32{1}, I32{2}, I32{5}, I32{8});
    const F32 cameraIndex = GENERATE(-3.0f, 0.0f, 1.5f, 3.9f, 7.0f, 20.0f);
    CAPTURE(count, cameraIndex);

    const I32 pivot = std::clamp(static_cast<I32>(std::floor(cameraIndex)), 0, count - 1);
    std::unordered_set<I32> visited;
    I32 previousBelow = -1;
    I32 previousAbove = count;
    for (I32 slot = 0; slot < count; ++slot) {
        const I32 cell = Modules::detail::Waterfall3DSweepOrder(slot, count, cameraIndex);
        REQUIRE(cell >= 0);
        REQUIRE(cell < count);
        REQUIRE(visited.insert(cell).second);
        if (cell < pivot) {
            REQUIRE(cell > previousBelow);
            previousBelow = cell;
        } else if (cell > pivot) {
            REQUIRE(cell < previousAbove);
            previousAbove = cell;
        } else {
            REQUIRE(slot == count - 1);
        }
    }
    REQUIRE(visited.size() == static_cast<U64>(count));
    REQUIRE(Modules::detail::Waterfall3DSweepOrder(count - 1, count, cameraIndex) == pivot);
}

TEST_CASE("Signal View surface camera clamps orbit, dolly, and pan",
          "[modules][signal_view][surface][camera]") {
    Modules::detail::Waterfall3DCamera camera;

    camera.orbit(0.0f, 10.0f);
    REQUIRE(camera.elevation == Catch::Approx(Modules::detail::kWaterfall3DMaxElevation));
    camera.orbit(0.0f, -10.0f);
    REQUIRE(camera.elevation == Catch::Approx(Modules::detail::kWaterfall3DMinElevation));

    camera.dolly(1000.0f);
    REQUIRE(camera.distance == Catch::Approx(Modules::detail::kWaterfall3DMaxDistance));
    camera.dolly(0.0001f);
    REQUIRE(camera.distance == Catch::Approx(Modules::detail::kWaterfall3DMinDistance));

    camera = {};
    camera.pan({50.0f, 0.0f}, 1.0f);
    REQUIRE(std::abs(camera.target.x) <= Modules::detail::kWaterfall3DPanLimit + 1e-5f);
    REQUIRE(std::abs(camera.target.z) <= Modules::detail::kWaterfall3DPanLimit + 1e-5f);
    camera.pan({0.0f, -50.0f}, 1.0f);
    REQUIRE(camera.target.y >= 0.0f);
    REQUIRE(camera.target.y <= Modules::detail::kWaterfall3DHeightScale);

    camera = {};
    const glm::vec3 eye = camera.eye();
    REQUIRE(glm::length(eye - camera.target) == Catch::Approx(camera.distance));
    REQUIRE(eye.y > camera.target.y);
    REQUIRE(eye.z > 0.0f);

    Modules::detail::Waterfall3DCamera goal;
    goal.azimuth = 2.0f;
    goal.distance = 5.0f;
    Modules::detail::Waterfall3DCamera follower;
    REQUIRE(follower.approach(goal, 0.5f));
    REQUIRE(follower.azimuth == Catch::Approx(0.5f * (0.55f + 2.0f)));
    for (int step = 0; step < 200; ++step) {
        follower.approach(goal, 0.5f);
    }
    REQUIRE_FALSE(follower.approach(goal, 0.5f));
    REQUIRE(follower.azimuth == Catch::Approx(goal.azimuth));
    REQUIRE(follower.distance == Catch::Approx(goal.distance));
}

TEST_CASE("Signal View surface hides the two walls nearest the camera",
          "[modules][signal_view][surface][scene]") {
    const auto corner = Modules::detail::Waterfall3DVisibleWalls({3.0f, 1.0f, 3.0f});
    REQUIRE(corner.negativeX);
    REQUIRE(corner.negativeZ);
    REQUIRE_FALSE(corner.positiveX);
    REQUIRE_FALSE(corner.positiveZ);

    const auto opposite = Modules::detail::Waterfall3DVisibleWalls({-3.0f, 1.0f, -3.0f});
    REQUIRE(opposite.positiveX);
    REQUIRE(opposite.positiveZ);
    REQUIRE_FALSE(opposite.negativeX);
    REQUIRE_FALSE(opposite.negativeZ);

    const auto inside = Modules::detail::Waterfall3DVisibleWalls({0.0f, 2.0f, 0.0f});
    REQUIRE(inside.negativeX);
    REQUIRE(inside.positiveX);
    REQUIRE(inside.negativeZ);
    REQUIRE(inside.positiveZ);
}

TEST_CASE("Signal View surface ticks label frequency, time, and amplitude",
          "[modules][signal_view][surface][labels]") {
    const auto frequency = Modules::detail::Waterfall3DFrequencyTicks(true, 100e6f, 10e6f);
    REQUIRE(frequency.size() == 5);
    REQUIRE(frequency.front().position == Catch::Approx(-1.0f));
    REQUIRE(frequency.front().label == "95.00");
    REQUIRE(frequency[2].label == "100.00");
    REQUIRE(frequency.back().label == "105.00");

    const auto normalized = Modules::detail::Waterfall3DFrequencyTicks(false, 0.0f, 0.0f);
    REQUIRE(normalized.front().label == "0.00");
    REQUIRE(normalized[2].label == "0.50");
    REQUIRE(normalized.back().label == "1.00");

    const auto time = Modules::detail::Waterfall3DTimeTicks(256);
    REQUIRE(time.size() == 5);
    REQUIRE(time.front().position == Catch::Approx(1.0f));
    REQUIRE(time.front().label == "0");
    REQUIRE(time[1].label == "-64");
    REQUIRE(time.back().position == Catch::Approx(-1.0f));
    REQUIRE(time.back().label == "-256");

    const auto amplitude = Modules::detail::Waterfall3DAmplitudeTicks(0.0f, 1.0f);
    REQUIRE(amplitude.size() == 5);
    REQUIRE(amplitude.front().position == Catch::Approx(0.0f));
    REQUIRE(amplitude.front().label == "0.00");
    REQUIRE(amplitude[1].label == "0.37");
    REQUIRE(amplitude[2].label == "0.50");
    REQUIRE(amplitude.back().position == Catch::Approx(1.0f));
    REQUIRE(amplitude.back().label == "1.00");
}

TEST_CASE("Signal View surface geometry builder caps capacity and clips behind the camera",
          "[modules][signal_view][surface][scene]") {
    using Modules::detail::kWaterfall3DGeometryStride;
    Modules::detail::Waterfall3DGeometry geometry(6);
    REQUIRE(geometry.capacity() == 6);
    geometry.line({-0.5f, 0.0f}, {0.5f, 0.0f}, {0.01f, 0.01f}, 2.0f, 1.0f);
    REQUIRE(geometry.used == 6);
    REQUIRE(geometry.storage[2] == Catch::Approx(2.0f));
    REQUIRE(geometry.storage[3] == Catch::Approx(1.0f));
    REQUIRE(geometry.storage[4] == Catch::Approx(1.0f));
    REQUIRE(geometry.storage[kWaterfall3DGeometryStride + 2] == Catch::Approx(-2.0f));
    REQUIRE(geometry.storage[1] == Catch::Approx(0.02f));
    geometry.line({-0.5f, 0.5f}, {0.5f, 0.5f}, {0.01f, 0.01f}, 2.0f, 1.0f);
    REQUIRE(geometry.used == 6);
    geometry.clear();
    geometry.quad({-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}, 0.5f);
    REQUIRE(geometry.used == 6);
    REQUIRE(geometry.storage[2] == Catch::Approx(0.0f));
    REQUIRE(geometry.storage[3] > 100.0f);
    REQUIRE(geometry.storage[4] == Catch::Approx(0.5f));

    Modules::detail::Waterfall3DCamera camera;
    const Modules::detail::Waterfall3DProjector projector{
        camera.projection(1.0f) * camera.view(), {0.01f, 0.01f}};
    REQUIRE(projector.project({0.0f, 0.0f, 0.0f}).has_value());
    REQUIRE_FALSE(projector.project(camera.eye() - camera.forward()).has_value());

    glm::vec2 a;
    glm::vec2 b;
    REQUIRE(projector.projectSegment({-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, a, b));
    REQUIRE(a.x < b.x);
    const glm::vec3 behind = camera.eye() - camera.forward() * 2.0f;
    REQUIRE_FALSE(projector.projectSegment(behind, behind - camera.forward(), a, b));
    REQUIRE(projector.projectSegment({0.0f, 0.0f, 0.0f}, behind, a, b));
    REQUIRE(std::isfinite(b.x));
    REQUIRE(std::isfinite(b.y));
}

TEST_CASE("Signal View surface axis frames, label extents, and tick thinning",
          "[modules][signal_view][surface][labels]") {
    const Modules::detail::Waterfall3DProjector projector{glm::mat4(1.0f), {0.01f, 0.01f}};
    const glm::vec2 center(0.0f, 0.0f);

    const auto floor = Modules::detail::Waterfall3DAxisFrameFor(
        projector, {-1.0f, -0.5f, 0.0f}, {1.0f, -0.5f, 0.0f}, {0.0f, -1.0f, 0.0f}, center);
    REQUIRE(floor.valid);
    REQUIRE(floor.direction.x == Catch::Approx(1.0f));
    REQUIRE(floor.normal.x == Catch::Approx(0.0f).margin(1e-6f));
    REQUIRE(floor.normal.y == Catch::Approx(-1.0f));

    const auto fallback = Modules::detail::Waterfall3DAxisFrameFor(
        projector, {-1.0f, -0.5f, 0.0f}, {1.0f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, center);
    REQUIRE(fallback.valid);
    REQUIRE(fallback.normal.y == Catch::Approx(-1.0f));

    const auto side = Modules::detail::Waterfall3DAxisFrameFor(
        projector, {-0.5f, -0.5f, 0.0f}, {-0.5f, 0.5f, 0.0f}, {-1.0f, 0.0f, 0.0f}, center);
    REQUIRE(side.valid);
    REQUIRE(side.normal.x == Catch::Approx(-1.0f));

    const auto degenerate = Modules::detail::Waterfall3DAxisFrameFor(
        projector, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, center);
    REQUIRE_FALSE(degenerate.valid);

    REQUIRE(Modules::detail::Waterfall3DLabelExtentAlong({0.0f, -1.0f}, {1, 0}, 30.0f, 12.0f) ==
            Catch::Approx(12.0f));
    REQUIRE(Modules::detail::Waterfall3DLabelExtentAlong({-1.0f, 0.0f}, {2, 1}, 30.0f, 12.0f) ==
            Catch::Approx(30.0f));
    REQUIRE(Modules::detail::Waterfall3DLabelExtentAlong({-1.0f, 0.0f}, {1, 1}, 30.0f, 12.0f) ==
            Catch::Approx(15.0f));
    REQUIRE(Modules::detail::Waterfall3DLabelExtentAlong({0.0f, 1.0f}, {0, 2}, 30.0f, 12.0f) ==
            Catch::Approx(12.0f));

    REQUIRE(Modules::detail::Waterfall3DTickStride(floor, projector.pixelSize, 5, 30.0f, 12.0f) == 1);
    REQUIRE(Modules::detail::Waterfall3DTickStride(floor, projector.pixelSize, 5, 60.0f, 12.0f) == 2);
    REQUIRE(Modules::detail::Waterfall3DTickStride(floor, projector.pixelSize, 5, 150.0f, 12.0f) == 4);
    REQUIRE(Modules::detail::Waterfall3DTickStride(side, projector.pixelSize, 5, 150.0f, 12.0f) == 1);
    REQUIRE(Modules::detail::Waterfall3DTickStride(side, projector.pixelSize, 5, 150.0f, 30.0f) == 2);
}

TEST_CASE("Signal View Surface decimates columns by peak",
          "[modules][signal_view][surface][decimate]") {
    REQUIRE(Modules::detail::Waterfall3DMeshColumns(64, 1000.0f) == 64);
    REQUIRE(Modules::detail::Waterfall3DMeshColumns(8192, 1000.0f) == 250);
    REQUIRE(Modules::detail::Waterfall3DMeshColumns(8192, 0.0f) ==
            Modules::detail::kWaterfall3DMinColumns);
    REQUIRE(Modules::detail::Waterfall3DMeshColumns(8192, 100000.0f) ==
            Modules::detail::kWaterfall3DMaxColumns);

    const std::vector<F32> identity = {0.1f, 0.4f, 0.2f, 0.9f};
    std::vector<F32> same(4, 0.0f);
    Modules::detail::Waterfall3DDecimateRow(identity.data(), 4, same.data(), 4);
    for (U64 i = 0; i < identity.size(); ++i) {
        REQUIRE(same[i] == Catch::Approx(identity[i]));
    }

    std::vector<F32> bins(9, 0.0f);
    bins[4] = 0.8f;
    std::vector<F32> reduced(3, 0.0f);
    Modules::detail::Waterfall3DDecimateRow(bins.data(), bins.size(), reduced.data(), 3);
    REQUIRE(reduced[0] == Catch::Approx(0.0f));
    REQUIRE(reduced[1] == Catch::Approx(0.8f));
    REQUIRE(reduced[2] == Catch::Approx(0.0f));
}
