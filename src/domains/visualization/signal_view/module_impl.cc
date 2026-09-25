#include "module_impl.hh"

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <span>

#include "jetstream/constants.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/tools/numeric.hh"
#include "resources/shaders/global_shaders.hh"
#include "resources/shaders/signal_view_shaders.hh"

namespace Jetstream::Modules {

namespace {

constexpr F32 kLineThickness = 1.0f;
constexpr ColorRGBA<F32> kAccentColor = {1.0f, 0.85f, 0.0f, 1.0f};
constexpr ColorRGBA<F32> kCursorLineColor = {1.0f, 1.0f, 1.0f, 0.35f};
constexpr ColorRGBA<F32> kCursorHaloColor = {0.0f, 0.0f, 0.0f, 0.65f};
constexpr ColorRGBA<F32> kCursorPillColor = {0.07f, 0.07f, 0.07f, 0.9f};
constexpr ColorRGBA<F32> kCursorPillEdgeColor = {1.0f, 1.0f, 1.0f, 0.22f};
constexpr std::array<ColorRGBA<F32>, detail::MaxMarkers> kMarkerPalette = {{
    {0.31f, 0.76f, 0.97f, 1.0f},
    {1.00f, 0.44f, 0.26f, 1.0f},
    {0.40f, 0.73f, 0.42f, 1.0f},
    {0.67f, 0.28f, 0.74f, 1.0f},
    {1.00f, 0.93f, 0.35f, 1.0f},
    {0.15f, 0.78f, 0.85f, 1.0f},
    {0.94f, 0.33f, 0.31f, 1.0f},
    {0.93f, 0.25f, 0.48f, 1.0f},
    {0.55f, 0.43f, 0.39f, 1.0f},
    {0.61f, 0.80f, 0.40f, 1.0f},
    {0.26f, 0.65f, 0.96f, 1.0f},
    {1.00f, 0.65f, 0.15f, 1.0f},
    {0.49f, 0.34f, 0.76f, 1.0f},
    {0.15f, 0.65f, 0.60f, 1.0f},
    {0.83f, 0.88f, 0.34f, 1.0f},
    {0.74f, 0.74f, 0.74f, 1.0f},
}};

constexpr ColorRGBA<F32> MarkerLineColor(const U64 index) {
    auto color = kMarkerPalette[index];
    color.a = 0.6f;
    return color;
}

constexpr ColorRGBA<F32> MarkerTagTextColor(const U64 index) {
    const auto& color = kMarkerPalette[index];
    const F32 luminance = 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
    return luminance > 0.5f ? ColorRGBA<F32>{0.05f, 0.05f, 0.05f, 1.0f}
                            : ColorRGBA<F32>{1.0f, 1.0f, 1.0f, 1.0f};
}
constexpr ColorRGBA<F32> kMarkerSpanColor = {1.0f, 1.0f, 1.0f, 0.6f};
constexpr F32 kLabelScale = 0.85f;
constexpr F32 kMarkerPickRadiusPx = 8.0f;
constexpr F32 kMarkerSpanThicknessPx = 3.0f;
constexpr F32 kMarkerSpanArrowArmPx = 11.0f;
constexpr F32 kMarkerSpanArrowAngleDeg = 30.0f;
constexpr F32 kMarkerSpanLabelGapPx = 10.0f;
constexpr F32 kMarkerSpanTagGapPx = 5.0f;
constexpr F32 kMarkerTableGapPx = 12.0f;

bool HostReadable(const Tensor& tensor) {
    return (static_cast<U8>(tensor.buffer().location()) &
            static_cast<U8>(Location::Host)) != 0 && tensor.data();
}

enum CursorInstance : U64 {
    kCursorLine = 0,
    kCursorHalo,
    kCursorMarker,
    kCursorPillEdge,
    kCursorPill,
    kCursorInstances,
};

enum MarkerGroup : U64 {
    kMarkerLine = 0,
    kMarkerHalo,
    kMarkerDot,
    kMarkerGroups,
};

enum MarkerTableGroup : U64 {
    kMarkerPillEdge = 0,
    kMarkerPill,
    kMarkerTableGroups,
};

enum MarkerSpanSegment : U64 {
    kSpanLeadSegment = 0,
    kSpanTrailSegment,
    kSpanSegments,
};

enum MarkerSpanArrow : U64 {
    kSpanLeadUpperArm = 0,
    kSpanLeadLowerArm,
    kSpanTrailUpperArm,
    kSpanTrailLowerArm,
    kSpanArrows,
};

constexpr U64 MarkerInstance(const U64 group, const U64 index) {
    return group * detail::MaxMarkers + index;
}

constexpr U64 SpanInstance(const U64 span, const U64 segment) {
    return span * kSpanSegments + segment;
}

constexpr U64 ArrowInstance(const U64 span, const U64 arm) {
    return span * kSpanArrows + arm;
}

std::string MarkerElement(const U64 index, const char* suffix) {
    return jst::fmt::format("marker-{}-{}", index, suffix);
}

std::string SpanElement(const U64 index, const char* suffix) {
    return jst::fmt::format("span-{}-{}", index, suffix);
}

}  // namespace

Result SignalViewImpl::validate() {
    const auto& config = *candidate();
    const bool hasLineplot = detail::SignalViewHasLineplot(config.mode);
    const bool hasWaterfall = detail::SignalViewHasWaterfall(config.mode);

    validatedNumberOfElements = 0;
    validatedNumberOfBatches = 0;
    validatedInputElementStride = 0;
    validatedInputBatchStride = 0;
    validatedNormalizationFactor = 0.0f;
    validatedLineplotEnabled = false;
    validatedWaterfallEnabled = false;

    if (!hasLineplot && !hasWaterfall) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (config.lineplotAveraging == 0) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Lineplot averaging must be at least 1.");
        return Result::ERROR;
    }

    if (config.waterfallAveraging == 0) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Waterfall averaging must be at least 1.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.rangeMin) || !std::isfinite(config.rangeMax)) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Display range must be finite.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.splitRatio) ||
        config.splitRatio < detail::MinSplitRatio || config.splitRatio > detail::MaxSplitRatio) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Split ratio must be between 0.1 and 0.9.");
        return Result::ERROR;
    }

    if (config.markers.size() > detail::MaxMarkers) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] At most {} markers are supported.", detail::MaxMarkers);
        return Result::ERROR;
    }
    for (const auto marker : config.markers) {
        if (!std::isfinite(marker) || marker < -1.0f || marker > 1.0f) {
            JST_ERROR("[MODULE_SIGNAL_VIEW] Marker positions must be between -1 and 1.");
            return Result::ERROR;
        }
    }
    for (const auto index : config.pins) {
        if (index >= config.markers.size()) {
            JST_ERROR("[MODULE_SIGNAL_VIEW] Pinned marker indices must refer to a marker.");
            return Result::ERROR;
        }
    }

    if (hasWaterfall &&
        (config.waterfallHeight == 0 || config.waterfallHeight > 8192)) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Invalid waterfall height value '{}', "
                  "must be between 1 and 8192.",
                  config.waterfallHeight);
        return Result::ERROR;
    }

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    SignalAxes axes;
    if (MapSignalAxes(inputTensor, IdentityAxisMap(inputTensor.rank()), axes) !=
        Result::SUCCESS) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Input must contain valid signal axis "
                  "metadata.");
        return Result::ERROR;
    }

    if (axes.sample && axes.channel) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Input cannot contain both sampleAxis "
                  "and channelAxis.");
        return Result::ERROR;
    }

    const auto elementAxis = axes.sample ? axes.sample : axes.channel;
    if (!elementAxis) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Input must contain sampleAxis or "
                  "channelAxis.");
        return Result::ERROR;
    }

    for (Index axis = 0; axis < inputTensor.rank(); ++axis) {
        if (axis != *elementAxis && (!axes.batch || axis != *axes.batch)) {
            JST_ERROR("[MODULE_SIGNAL_VIEW] Unsupported auxiliary input axis {}. "
                      "Every dimension must be the element axis or batchAxis.",
                      axis);
            return Result::ERROR;
        }
    }

    const U64 numberOfElements = inputTensor.shape(*elementAxis);
    if (hasLineplot && numberOfElements < 2) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Invalid number of elements ({}), need "
                  "at least 2.",
                  numberOfElements);
        return Result::ERROR;
    }

    if (hasLineplot) {
        const U64 maxRenderScalarCount =
            std::min(static_cast<U64>(std::numeric_limits<std::size_t>::max()) /
                         sizeof(F32),
                     static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) /
                         sizeof(F32));
        U64 signalPointScalarCount = 0;
        U64 signalVertexScalarCount = 0;
        U64 signalVertexCount = 0;
        U64 fillVertexScalarCount = 0;
        if (!Jetstream::detail::CheckedMultiply(numberOfElements, 2,
                                                signalPointScalarCount) ||
            !Jetstream::detail::CheckedMultiply(numberOfElements - 1, 16,
                                                signalVertexScalarCount) ||
            !Jetstream::detail::CheckedMultiply(numberOfElements - 1, 4,
                                                signalVertexCount) ||
            !Jetstream::detail::CheckedMultiply(numberOfElements, 4,
                                                fillVertexScalarCount) ||
            numberOfElements > std::numeric_limits<U32>::max() ||
            signalVertexCount > std::numeric_limits<U32>::max() ||
            signalPointScalarCount > maxRenderScalarCount ||
            signalVertexScalarCount > maxRenderScalarCount ||
            fillVertexScalarCount > maxRenderScalarCount) {
            JST_ERROR("[MODULE_SIGNAL_VIEW] Line geometry exceeds the supported "
                      "rendering range.");
            return Result::ERROR;
        }
    }

    if (hasWaterfall) {
        U64 waterfallElementCount = 0;
        const U64 maxWaterfallElementCount = std::min({
            static_cast<U64>(std::numeric_limits<I32>::max()),
            static_cast<U64>(std::numeric_limits<std::size_t>::max()) /
                sizeof(F32),
            static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) /
                sizeof(F32),
        });
        if (!Jetstream::detail::CheckedMultiply(numberOfElements,
                                                config.waterfallHeight,
                                                waterfallElementCount) ||
            waterfallElementCount > maxWaterfallElementCount) {
            JST_ERROR("[MODULE_SIGNAL_VIEW] Waterfall geometry exceeds the "
                      "supported rendering range.");
            return Result::ERROR;
        }
    }

    for (const auto* attribute : {"frequency", "sampleRate"}) {
        if (!inputTensor.hasAttribute(attribute)) {
            continue;
        }
        const std::any value = inputTensor.attribute(attribute);
        const auto* scalar = std::any_cast<F32>(&value);
        if (!scalar || !std::isfinite(*scalar)) {
            JST_ERROR("[MODULE_SIGNAL_VIEW] Input {} metadata must be a finite F32.",
                      attribute);
            return Result::ERROR;
        }
    }

    const U64 numberOfBatches = axes.batch
        ? inputTensor.shape(*axes.batch)
        : 1;
    validatedNumberOfElements = numberOfElements;
    validatedNumberOfBatches = numberOfBatches;
    validatedInputElementStride = inputTensor.stride(*elementAxis);
    validatedInputBatchStride = axes.batch
        ? inputTensor.stride(*axes.batch)
        : 0;
    validatedNormalizationFactor =
        1.0f / (0.5f * static_cast<F32>(numberOfBatches));
    validatedLineplotEnabled = hasLineplot;
    validatedWaterfallEnabled = hasWaterfall;

    return Result::SUCCESS;
}

Result SignalViewImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("signal"));

    return Result::SUCCESS;
}

Result SignalViewImpl::create() {
    splitter = {};
    splitter.ratio = splitRatio;
    updateLayoutFlag = false;
    displayHeld = false;
    cursor = {};
    markerPositions = markers;
    updateMarkersFlag = false;
    applyPins();
    displayedPoints.clear();

    // Get input tensor.

    input = inputs().at("signal").tensor;

    if (input.hasAttribute("frequency")) {
        const std::any value = input.attribute("frequency");
        if (const auto* frequency = std::any_cast<F32>(&value)) {
            JST_DEBUG("[MODULE_SIGNAL_VIEW] Input frequency: {:.02f} MHz",
                      *frequency / 1e6f);
        }
    }
    if (input.hasAttribute("sampleRate")) {
        const std::any value = input.attribute("sampleRate");
        if (const auto* sampleRate = std::any_cast<F32>(&value)) {
            JST_DEBUG("[MODULE_SIGNAL_VIEW] Input sample rate: {:.02f} MHz",
                      *sampleRate / 1e6f);
        }
    }

    numberOfElements = validatedNumberOfElements;
    numberOfBatches = validatedNumberOfBatches;
    inputElementStride = validatedInputElementStride;
    inputBatchStride = validatedInputBatchStride;
    normalizationFactor = validatedNormalizationFactor;
    lineplotEnabled = validatedLineplotEnabled;
    waterfallEnabled = validatedWaterfallEnabled;
    waterfallAveragingCount = 0;
    maxHoldWarmupBlocks = 0;
    lineplotAveragingInitialized = false;
    waterfallHistory = {};
    updateSignalPointsFlag = false;
    updateHoldPointsFlag = false;
    updateSignalUniformBufferFlag = false;

    const Buffer::Config renderStateConfig = renderStateBufferConfig();

    if (lineplotEnabled) {
        JST_CHECK(signalPoints.create(device(), DataType::F32,
                                      {numberOfElements, 2},
                                      renderStateConfig));
        JST_CHECK(signalVertices.create(device(), DataType::F32,
                                        {numberOfElements - 1, 4, 4},
                                        renderStateConfig));
        if (fill) {
            JST_CHECK(fillVertices.create(device(), DataType::F32,
                                          {numberOfElements * 2, 2},
                                          renderStateConfig));
        }

        JST_CHECK(maxHoldPoints.create(device(), DataType::F32,
                                       {numberOfElements, 2},
                                       renderStateConfig));
        JST_CHECK(maxHoldVertices.create(device(), DataType::F32,
                                         {numberOfElements - 1, 4, 4},
                                         renderStateConfig));
    }

    if (waterfallEnabled) {
        JST_CHECK(waterfallBins.create(device(), DataType::F32,
                                       {waterfallHeight, numberOfElements},
                                       renderStateConfig));
    }

    return Result::SUCCESS;
}

Result SignalViewImpl::destroy() {
    JST_CHECK(destroyPresent());
    return Result::SUCCESS;
}

Result SignalViewImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.mode == mode &&
        config.maxHold == maxHold &&
        config.fill == fill &&
        config.waterfallHeight == waterfallHeight &&
        config.xLabel == xLabel &&
        config.amplitudeLabel == amplitudeLabel &&
        config.waterfallLabel == waterfallLabel) {
        const bool lineplotAveragingChanged = config.lineplotAveraging != lineplotAveraging;
        const bool waterfallAveragingChanged = config.waterfallAveraging != waterfallAveraging;
        const bool rangeChanged =
            config.rangeMin != rangeMin || config.rangeMax != rangeMax;
        updateLayoutFlag |= config.splitRatio != splitRatio;
        splitRatio = config.splitRatio;
        updateMarkersFlag |= config.markers != markers || config.pins != pins;
        markers = config.markers;
        pins = config.pins;
        lineplotAveraging = config.lineplotAveraging;
        waterfallAveraging = config.waterfallAveraging;
        rangeMin = config.rangeMin;
        rangeMax = config.rangeMax;
        if (lineplotAveragingChanged) {
            JST_CHECK(resetLineplotHistory());
        }
        if (waterfallAveragingChanged) {
            waterfallAveragingCount = 0;
        }
        if (rangeChanged) {
            JST_CHECK(resetHistoryState());
        }
        return Result::SUCCESS;
    }

    return Result::RECREATE;
}

Result SignalViewImpl::resetLineplotHistory() {
    if (lineplotEnabled) {
        F32* maxData = static_cast<F32*>(maxHoldPoints.data());
        for (U64 i = 0; i < numberOfElements; i++) {
            maxData[(i * 2) + 1] = -1.0f;
        }
        maxHoldWarmupBlocks = 0;
        updateHoldPointsFlag = true;
    }

    lineplotAveragingInitialized = false;
    return Result::SUCCESS;
}

Result SignalViewImpl::resetHistoryState() {
    JST_CHECK(resetLineplotHistory());

    if (waterfallEnabled) {
        std::fill(static_cast<F32*>(waterfallBins.data()),
                  static_cast<F32*>(waterfallBins.data()) +
                      waterfallBins.size(),
                  0.0f);
        waterfallHistory = {};
        waterfallHistory.dirtyRows = waterfallHeight;
        waterfallAveragingCount = 0;
    }

    return Result::SUCCESS;
}

Result SignalViewImpl::createPresent() {
    auto& window = render();

    if (!window) {
        JST_DEBUG("[MODULE_SIGNAL_VIEW] No render window available, skipping "
                  "present creation.");
        return Result::SUCCESS;
    }

    JST_DEBUG("[MODULE_SIGNAL_VIEW] Creating present resources...");

    // Axis component.

    if (!window->hasFont("default_mono")) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Font 'default_mono' not found.");
        return Result::ERROR;
    }

    {
        Render::Components::Axis::Config cfg;
        cfg.thickness = kLineThickness;
        cfg.showInteriorGrid = lineplotEnabled;
        const bool combined = lineplotEnabled && waterfallEnabled;
        cfg.verticalScale = combined ? splitRatio : 1.0f;
        cfg.showFrameTicks = lineplotEnabled;
        cfg.font = window->font("default_mono");
        cfg.xTitle = xLabel;
        cfg.yTitle = combined ? "" : (lineplotEnabled
            ? amplitudeLabel
            : waterfallLabel);
        cfg.yLabelOnRight = lineplotEnabled;
        cfg.gridColor = {0.12f, 0.12f, 0.12f, 1.0f};
        cfg.majorGridColor = {0.5f, 0.5f, 0.5f, 1.0f};
        JST_CHECK(window->build(axis, cfg));
        JST_CHECK(window->bind(axis));
    }

    // Lineplot layer.

    if (lineplotEnabled) {
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &signalUniforms;
            cfg.elementByteSize = sizeof(signalUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(signalUniformBuffer, cfg));
        }

        auto buildTrace = [&](
            const std::shared_ptr<Render::Buffer>& uniformBuffer,
            Tensor& pointsTensor,
            std::shared_ptr<Render::Buffer>& pointsBuffer,
            Tensor& verticesTensor,
            std::shared_ptr<Render::Buffer>& verticesBuffer,
            std::shared_ptr<Render::Kernel>& kernel,
            std::shared_ptr<Render::Vertex>& vertex,
            std::shared_ptr<Render::Draw>& draw,
            std::shared_ptr<Render::Program>& program) -> Result {
            {
                Render::Buffer::Config cfg;
                cfg.buffer = pointsTensor.data();
                cfg.elementByteSize = sizeof(F32);
                cfg.size = pointsTensor.size();
                cfg.target = Render::Buffer::Target::STORAGE;
                cfg.enableZeroCopy = false;
                JST_CHECK(window->build(pointsBuffer, cfg));
            }
            {
                Render::Buffer::Config cfg;
                cfg.buffer = verticesTensor.data();
                cfg.elementByteSize = sizeof(F32);
                cfg.size = verticesTensor.size();
                cfg.target = Render::Buffer::Target::VERTEX |
                             Render::Buffer::Target::STORAGE;
                cfg.enableZeroCopy = false;
                JST_CHECK(window->build(verticesBuffer, cfg));
            }
            {
                Render::Kernel::Config cfg;
                cfg.gridSize = {numberOfElements - 1, 1, 1};
                cfg.kernels = GlobalKernelsPackage["thicklinestrip"];
                cfg.buffers = {
                    {uniformBuffer, Render::Kernel::AccessMode::READ},
                    {pointsBuffer, Render::Kernel::AccessMode::READ},
                    {verticesBuffer, Render::Kernel::AccessMode::WRITE},
                };
                JST_CHECK(window->build(kernel, cfg));
            }
            {
                Render::Vertex::Config cfg;
                cfg.vertices = {
                    {verticesBuffer, 4},
                };
                JST_CHECK(window->build(vertex, cfg));
            }
            {
                Render::Draw::Config cfg;
                cfg.buffer = vertex;
                cfg.mode = Render::Draw::Mode::TRIANGLE_STRIP;
                JST_CHECK(window->build(draw, cfg));
            }
            {
                Render::Program::Config cfg;
                cfg.shaders = ShadersPackage["signal"];
                cfg.draws = {draw};
                cfg.buffers = {
                    {uniformBuffer,
                     Render::Program::Target::VERTEX |
                         Render::Program::Target::FRAGMENT},
                };
                cfg.enableAlphaBlending = true;
                JST_CHECK(window->build(program, cfg));
            }
            return Result::SUCCESS;
        };

        JST_CHECK(buildTrace(signalUniformBuffer,
                             signalPoints, signalPointsBuffer,
                             signalVertices, signalVerticesBuffer,
                             signalKernel, signalVertex,
                             drawSignalVertex, signalProgram));

        // Fill element (analyser-style persistence area beneath the trace).

        if (fill) {
            {
                Render::Buffer::Config cfg;
                cfg.buffer = fillVertices.data();
                cfg.elementByteSize = sizeof(F32);
                cfg.size = fillVertices.size();
                cfg.target = Render::Buffer::Target::VERTEX |
                             Render::Buffer::Target::STORAGE;
                cfg.enableZeroCopy = false;
                JST_CHECK(window->build(fillVerticesBuffer, cfg));
            }

            {
                Render::Kernel::Config cfg;
                cfg.gridSize = {numberOfElements, 1, 1};
                cfg.kernels = GlobalKernelsPackage["fillarea"];
                cfg.buffers = {
                    {signalUniformBuffer, Render::Kernel::AccessMode::READ},
                    {signalPointsBuffer, Render::Kernel::AccessMode::READ},
                    {fillVerticesBuffer, Render::Kernel::AccessMode::WRITE},
                };
                JST_CHECK(window->build(fillKernel, cfg));
            }

            {
                Render::Vertex::Config cfg;
                cfg.vertices = {
                    {fillVerticesBuffer, 2},
                };
                JST_CHECK(window->build(fillVertex, cfg));
            }

            {
                Render::Draw::Config cfg;
                cfg.buffer = fillVertex;
                cfg.mode = Render::Draw::Mode::TRIANGLE_STRIP;
                JST_CHECK(window->build(drawFillVertex, cfg));
            }

            {
                Render::Program::Config cfg;
                cfg.shaders = ShadersPackage["fill"];
                cfg.draws = {drawFillVertex};
                cfg.buffers = {
                    {signalUniformBuffer,
                     Render::Program::Target::VERTEX |
                         Render::Program::Target::FRAGMENT},
                };
                cfg.enableAlphaBlending = true;
                JST_CHECK(window->build(fillProgram, cfg));
            }
        }

        // Max hold trace (dimmed grey line behind the live trace).

        {
            Render::Buffer::Config cfg;
            cfg.buffer = &holdUniforms;
            cfg.elementByteSize = sizeof(holdUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(holdUniformBuffer, cfg));
        }

        JST_CHECK(buildTrace(holdUniformBuffer,
                             maxHoldPoints, maxHoldPointsBuffer,
                             maxHoldVertices, maxHoldVerticesBuffer,
                             maxHoldKernel, maxHoldVertex,
                             drawMaxHoldVertex, maxHoldProgram));
    }

    if (waterfallEnabled) {
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &FillScreenVertices;
            cfg.elementByteSize = sizeof(F32);
            cfg.size = 12;
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(fillScreenVerticesBuffer, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = &FillScreenTextureVertices;
            cfg.elementByteSize = sizeof(F32);
            cfg.size = 8;
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(fillScreenTextureVerticesBuffer, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = &FillScreenIndices;
            cfg.elementByteSize = sizeof(U32);
            cfg.size = 6;
            cfg.target = Render::Buffer::Target::VERTEX_INDICES;
            JST_CHECK(window->build(fillScreenIndicesBuffer, cfg));
        }

        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {fillScreenVerticesBuffer, 3},
                {fillScreenTextureVerticesBuffer, 2},
            };
            cfg.indices = fillScreenIndicesBuffer;
            JST_CHECK(window->build(waterfallVertex, cfg));
        }

        {
            Render::Draw::Config cfg;
            cfg.buffer = waterfallVertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            JST_CHECK(window->build(drawWaterfallVertex, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = waterfallBins.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = waterfallBins.size();
            cfg.target = Render::Buffer::Target::STORAGE;
            cfg.enableZeroCopy = false;
            JST_CHECK(window->build(waterfallBuffer, cfg));
        }

        {
            Render::Texture::Config cfg;
            cfg.size = {256, 1};
            cfg.buffer = const_cast<U8*>(&TurboLutBytes[0][0]);
            JST_CHECK(window->build(waterfallLutTexture, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = &waterfallUniforms;
            cfg.elementByteSize = sizeof(waterfallUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(waterfallUniformBuffer, cfg));
        }

        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["waterfall"];
            cfg.draws = {drawWaterfallVertex};
            cfg.textures = {waterfallLutTexture};
            cfg.buffers = {
                {waterfallUniformBuffer, Render::Program::Target::VERTEX |
                                         Render::Program::Target::FRAGMENT},
                {waterfallBuffer, Render::Program::Target::FRAGMENT},
            };
            JST_CHECK(window->build(waterfallProgram, cfg));
        }
    }

    // Text labels (header, zoom, and cursor readouts).

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 256;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        cfg.elements = {
            {"header",
             {.scale = kLabelScale,
              .position = {-1.0f, 1.0f},
              .alignment = {0, 0}}},
            {"zoom",
             {.scale = kLabelScale,
              .position = {0.0f, 1.0f},
              .alignment = {1, 0}}},
            {"hold",
             {.scale = kLabelScale,
              .position = {-1.0f, 1.0f},
              .alignment = {0, 0},
              .color = kAccentColor}},
            {"amplitude-title",
             {.scale = kLabelScale,
              .position = {-1.0f, 0.5f},
              .alignment = {1, 0},
              .rotationDeg = 90.0f}},
            {"waterfall-title",
             {.scale = kLabelScale,
              .position = {-1.0f, -0.5f},
              .alignment = {1, 0},
              .rotationDeg = 90.0f}},
        };
        JST_CHECK(window->build(text, cfg));
        JST_CHECK(window->bind(text));
    }

    // Cursor overlay (line, trace marker, and readout pill).

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["cursor"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kCursorInstances,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 1.0e4f,
        };
        JST_CHECK(window->build(cursorShapes, cfg));
        JST_CHECK(window->bind(cursorShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(cursorShapes->getColors("cursor", colors));
        colors[kCursorLine] = kCursorLineColor;
        colors[kCursorHalo] = kCursorHaloColor;
        colors[kCursorMarker] = kAccentColor;
        colors[kCursorPillEdge] = kCursorPillEdgeColor;
        colors[kCursorPill] = kCursorPillColor;
        JST_CHECK(cursorShapes->updateColors("cursor"));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 64;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        cfg.elements = {
            {"cursor-x",
             {.scale = kLabelScale,
              .position = {-2.0f, -2.0f},
              .alignment = {0, 1}}},
            {"cursor-y",
             {.scale = kLabelScale,
              .position = {-2.0f, -2.0f},
              .alignment = {0, 1},
              .color = kAccentColor}},
        };
        JST_CHECK(window->build(cursorText, cfg));
        JST_CHECK(window->bind(cursorText));
    }

    // Marker overlay (lines, trace dots, and readout table).

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["markers"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kMarkerGroups * detail::MaxMarkers,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 1.0e4f,
        };
        JST_CHECK(window->build(markerShapes, cfg));
        JST_CHECK(window->bind(markerShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(markerShapes->getColors("markers", colors));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            colors[MarkerInstance(kMarkerLine, i)] = MarkerLineColor(i);
            colors[MarkerInstance(kMarkerHalo, i)] = kCursorHaloColor;
            colors[MarkerInstance(kMarkerDot, i)] = kMarkerPalette[i];
        }
        JST_CHECK(markerShapes->updateColors("markers"));
    }

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["tags"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = detail::MaxMarkers,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 4.0f,
        };
        JST_CHECK(window->build(markerTagShapes, cfg));
        JST_CHECK(window->bind(markerTagShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(markerTagShapes->getColors("tags", colors));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            colors[i] = kMarkerPalette[i];
        }
        JST_CHECK(markerTagShapes->updateColors("tags"));
    }

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["spans"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kSpanSegments * detail::MarkerSpans,
            .color = kMarkerSpanColor,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = kMarkerSpanThicknessPx * 0.5f,
        };
        cfg.elements["arrows"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kSpanArrows * detail::MarkerSpans,
            .color = kMarkerSpanColor,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = kMarkerSpanThicknessPx * 0.5f,
        };
        JST_CHECK(window->build(markerSpanShapes, cfg));
        JST_CHECK(window->bind(markerSpanShapes));
    }

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["table"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = kMarkerTableGroups * detail::MaxMarkers,
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .cornerRadius = 1.0e4f,
        };
        JST_CHECK(window->build(markerTableShapes, cfg));
        JST_CHECK(window->bind(markerTableShapes));

        std::span<ColorRGBA<F32>> colors;
        JST_CHECK(markerTableShapes->getColors("table", colors));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            colors[MarkerInstance(kMarkerPillEdge, i)] = kCursorPillEdgeColor;
            colors[MarkerInstance(kMarkerPill, i)] = kCursorPillColor;
        }
        JST_CHECK(markerTableShapes->updateColors("table"));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 1024;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            cfg.elements[MarkerElement(i, "x")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {0, 1},
            };
            cfg.elements[MarkerElement(i, "y")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {0, 1},
            };
        }
        for (U64 i = 0; i < detail::MarkerSpans; ++i) {
            cfg.elements[SpanElement(i, "label")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {1, 1},
                .color = kMarkerSpanColor,
            };
        }
        JST_CHECK(window->build(markerText, cfg));
        JST_CHECK(window->bind(markerText));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 64;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->hasFont("default_mono_bold")
            ? window->font("default_mono_bold")
            : window->font("default_mono");
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            cfg.elements[MarkerElement(i, "id")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {0, 1},
                .color = kMarkerPalette[i],
            };
        }
        JST_CHECK(window->build(markerBadgeText, cfg));
        JST_CHECK(window->bind(markerBadgeText));
    }

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 64;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->hasFont("default_mono_bold")
            ? window->font("default_mono_bold")
            : window->font("default_mono");
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            cfg.elements[MarkerElement(i, "tag")] = {
                .scale = kLabelScale,
                .position = {-2.0f, -2.0f},
                .alignment = {1, 1},
                .color = MarkerTagTextColor(i),
            };
        }
        JST_CHECK(window->build(markerTagText, cfg));
        JST_CHECK(window->bind(markerTagText));
    }

    // Framebuffer texture.

    {
        Render::Texture::Config cfg;
        cfg.size = interaction.viewSize;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }

    // Surface.

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.multisampled = lineplotEnabled;
        cfg.clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        if (waterfallEnabled) {
            cfg.programs.push_back(waterfallProgram);
        }
        JST_CHECK(axis->surfaceUnderlay(cfg));
        if (lineplotEnabled) {
            cfg.kernels.push_back(signalKernel);
            if (fill) {
                cfg.kernels.push_back(fillKernel);
            }
            if (maxHold) {
                cfg.kernels.push_back(maxHoldKernel);
            }
            if (fill) {
                cfg.programs.push_back(fillProgram);
            }
            if (maxHold) {
                cfg.programs.push_back(maxHoldProgram);
            }
            cfg.programs.push_back(signalProgram);
        }
        JST_CHECK(markerShapes->surface(cfg));
        JST_CHECK(markerSpanShapes->surface(cfg));
        JST_CHECK(markerTagShapes->surface(cfg));
        JST_CHECK(markerTagText->surface(cfg));
        JST_CHECK(markerTableShapes->surface(cfg));
        JST_CHECK(markerText->surface(cfg));
        JST_CHECK(markerBadgeText->surface(cfg));
        JST_CHECK(axis->surfaceOverlay(cfg));
        JST_CHECK(text->surface(cfg));
        JST_CHECK(cursorShapes->surface(cfg));
        JST_CHECK(cursorText->surface(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    if (lineplotEnabled) {
        signalUniforms.traceColor[0] = 1.0f;
        signalUniforms.traceColor[1] = 0.85f;
        signalUniforms.traceColor[2] = 0.0f;
        signalUniforms.traceColor[3] = 0.25f;

        holdUniforms.traceColor[0] = 0.6f;
        holdUniforms.traceColor[1] = 0.45f;
        holdUniforms.traceColor[2] = 0.0f;
        holdUniforms.traceColor[3] = 0.7f;
    }

    updateState();

    // Register surface manifest.

    JST_CHECK(surfaceCreateManifest({
        .id = "default",
        .size = interaction.viewSize,
        .surface = framebufferTexture,
    }));

    return Result::SUCCESS;
}

Result SignalViewImpl::destroyPresent() {
    auto& window = render();

    if (!window) {
        return Result::SUCCESS;
    }

    if (renderSurface) {
        JST_CHECK(window->unbind(renderSurface));
    }
    if (cursorText) {
        JST_CHECK(window->unbind(cursorText));
    }
    if (cursorShapes) {
        JST_CHECK(window->unbind(cursorShapes));
    }
    if (markerTagText) {
        JST_CHECK(window->unbind(markerTagText));
    }
    if (markerBadgeText) {
        JST_CHECK(window->unbind(markerBadgeText));
    }
    if (markerText) {
        JST_CHECK(window->unbind(markerText));
    }
    if (markerTableShapes) {
        JST_CHECK(window->unbind(markerTableShapes));
    }
    if (markerTagShapes) {
        JST_CHECK(window->unbind(markerTagShapes));
    }
    if (markerSpanShapes) {
        JST_CHECK(window->unbind(markerSpanShapes));
    }
    if (markerShapes) {
        JST_CHECK(window->unbind(markerShapes));
    }
    if (text) {
        JST_CHECK(window->unbind(text));
    }
    if (axis) {
        JST_CHECK(window->unbind(axis));
    }
    return Result::SUCCESS;
}

Result SignalViewImpl::present() {
    if (!renderSurface) {
        return Result::SUCCESS;
    }

    // Process surface interaction events.

    interaction = ProcessSurfaceInteraction(interaction,
                                            surfaceConsumeSurfaceEvents(), {});
    // Resize the axis before hit-testing so input and rendering use the same
    // padded plot rectangle, including on a resize-and-click frame.
    JST_CHECK(axis->updatePixelSize({
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y,
    }));
    processInputEvents(axis->paddingScale());

    if (interaction.viewChanged || updateLayoutFlag) {
        renderSurface->size(interaction.viewSize);
        renderSurface->clearColor(interaction.backgroundColor);
        surfaceUpdateManifestSize("default", interaction.viewSize);
        updateState();
        updateLayoutFlag = false;
    }

    if (waterfallEnabled) {
        if (!displayHeld) {
            const auto dirtyPlan = waterfallHistory.dirtyPlan(waterfallHeight);
            if (dirtyPlan.firstRowCount > 0) {
                JST_CHECK(waterfallBuffer->update(dirtyPlan.startRow *
                                                      numberOfElements,
                                                  dirtyPlan.firstRowCount *
                                                      numberOfElements));
            }
            if (dirtyPlan.secondRowCount > 0) {
                JST_CHECK(waterfallBuffer->update(0,
                                                  dirtyPlan.secondRowCount *
                                                      numberOfElements));
            }
            waterfallHistory.clearDirty();
            waterfallUniforms.index = waterfallHistory.writeIndex /
                                      static_cast<F32>(waterfallHeight);
        }

        waterfallUniforms.width = static_cast<int>(numberOfElements);
        waterfallUniforms.height = static_cast<int>(waterfallHeight);
        waterfallUniforms.offset = interaction.offset +
            0.5f * (1.0f - 1.0f / interaction.zoom);
        waterfallUniforms.zoom = interaction.zoom;
        JST_CHECK(waterfallUniformBuffer->update());
    }

    // Process update flags.

    if (lineplotEnabled && updateSignalPointsFlag && !displayHeld) {
        JST_CHECK(signalPointsBuffer->update());
        if (HostReadable(signalPoints)) {
            const F32* points = signalPoints.data<F32>();
            displayedPoints.assign(points, points + signalPoints.size());
        }
        signalKernel->update();
        if (fill) {
            fillKernel->update();
        }
        if (maxHold) {
            if (updateHoldPointsFlag) {
                JST_CHECK(maxHoldPointsBuffer->update());
                maxHoldKernel->update();
                updateHoldPointsFlag = false;
            }
        }
        updateSignalPointsFlag = false;
    }

    if (lineplotEnabled && updateSignalUniformBufferFlag) {
        JST_CHECK(signalUniformBuffer->update());
        signalKernel->update();
        if (fill) {
            fillKernel->update();
        }
        if (maxHold) {
            JST_CHECK(holdUniformBuffer->update());
            maxHoldKernel->update();
        }
        updateSignalUniformBufferFlag = false;
    }

    updateLabelState();
    JST_CHECK(updateMarkerState());
    JST_CHECK(updateCursorState());

    JST_CHECK(axis->present());
    if (text) {
        JST_CHECK(text->present());
    }
    if (markerShapes) {
        JST_CHECK(markerShapes->present());
    }
    if (markerSpanShapes) {
        JST_CHECK(markerSpanShapes->present());
    }
    if (markerTagShapes) {
        JST_CHECK(markerTagShapes->present());
    }
    if (markerTableShapes) {
        JST_CHECK(markerTableShapes->present());
    }
    if (markerText) {
        JST_CHECK(markerText->present());
    }
    if (markerBadgeText) {
        JST_CHECK(markerBadgeText->present());
    }
    if (markerTagText) {
        JST_CHECK(markerTagText->present());
    }
    if (cursorShapes) {
        JST_CHECK(cursorShapes->present());
    }
    if (cursorText) {
        JST_CHECK(cursorText->present());
    }

    return Result::SUCCESS;
}

void SignalViewImpl::processInputEvents(const Extent2D<F32>& paddingScale) {
    const F32 previousRatio = splitter.ratio;
    if (!splitter.dragging && !configChangePending()) {
        splitter.ratio = splitRatio;
    }

    syncMarkers();

    std::vector<InputEvent> plotEvents;
    const bool enabled = lineplotEnabled && waterfallEnabled &&
                         configChangeEnabled("splitRatio");
    for (const auto& input : surfaceConsumeInputEvents()) {
        if (const auto* key = std::get_if<KeyEvent>(&input)) {
            if (key->type == KeyEventType::Press && key->key == KeyCode::Space && !key->repeat) {
                displayHeld = !displayHeld;
            }
            if (key->type == KeyEventType::Press && key->key == KeyCode::Escape &&
                std::any_of(pinned.begin(), pinned.end(), [](const bool flag) { return flag; })) {
                pinned.fill(false);
                commitMarkers();
            }
        }
        const auto mouse = SurfaceMouseEvent(input);
        if (!mouse) continue;
        const auto& event = *mouse;
        if (event.type == MouseEventType::Move) {
            cursor.inside = true;
            cursor.position = event.position;
        } else if (event.type == MouseEventType::Leave) {
            cursor.inside = false;
        }
        if (event.type == MouseEventType::Click && event.modifiers.shift && !splitter.dragging) {
            cursor.inside = true;
            cursor.position = event.position;
            if (event.button == MouseButton::Left) {
                toggleMarker();
            } else if (event.button == MouseButton::Right) {
                clearMarkers();
            }
            continue;
        }
        if (markerDrag.index && *markerDrag.index >= markerPositions.size()) {
            markerDrag = {};
        }
        if (markerDrag.index) {
            const bool released = event.type == MouseEventType::Release &&
                                  event.button == MouseButton::Left;
            if (event.type == MouseEventType::Move || released) {
                cursor.inside = true;
                cursor.position = event.position;
                const F32 travel = std::abs(event.position.x - markerDrag.origin.x) *
                                   static_cast<F32>(interaction.viewSize.x);
                if (markerDrag.moved || travel > 3.0f * interaction.scale) {
                    markerDrag.moved = true;
                    markerPositions[*markerDrag.index] = pointAtX(event.position.x);
                }
                if (!released) {
                    continue;
                }
            }
            if (released || event.type == MouseEventType::Leave) {
                if (event.type == MouseEventType::Leave) {
                    cursor.inside = false;
                }
                if (markerDrag.moved) {
                    commitMarkers();
                }
                markerDrag = {};
                continue;
            }
        }
        if (event.type == MouseEventType::Click && event.button == MouseButton::Left &&
            !splitter.dragging) {
            cursor.inside = true;
            cursor.position = event.position;
            if (const auto tag = tagAt(event.position)) {
                pinned[*tag] = !pinned[*tag];
                commitMarkers();
                continue;
            }
            if (const auto hit = markerAt(event.position)) {
                markerDrag = {.index = hit, .origin = event.position};
                continue;
            }
        }
        const auto layout = detail::CalculateSignalViewPanels(paddingScale,
                                                               interaction.viewSize,
                                                               splitter.ratio);
        bool commit = false;
        if (splitter.process(event, layout, interaction.viewSize,
                              interaction.scale, enabled, commit)) {
            // Splitter capture must never start or continue a horizontal pan.
            interaction.dragging = false;
            if (commit && (splitter.ratio != splitRatio || configChangePending())) {
                Parser::Map edit;
                edit["splitRatio"] = splitter.ratio;
                if (requestConfigChange(edit) != Result::SUCCESS) {
                    splitter.ratio = splitRatio;
                }
            } else if (event.type == MouseEventType::Leave) {
                splitter.ratio = splitRatio;
            }
        } else {
            plotEvents.push_back(event);
        }
    }
    const bool viewChanged = interaction.viewChanged || previousRatio != splitter.ratio;
    interaction = ProcessSurfaceInteraction(interaction, {}, std::move(plotEvents));
    interaction.viewChanged |= viewChanged;

    SurfaceCursor shape = SurfaceCursor::Default;
    if (markerDrag.index) {
        shape = SurfaceCursor::ResizeEW;
    } else if (splitter.dragging) {
        shape = SurfaceCursor::ResizeNS;
    } else if (cursor.inside) {
        const auto layout = detail::CalculateSignalViewPanels(paddingScale,
                                                               interaction.viewSize,
                                                               splitter.ratio);
        if (tagAt(cursor.position)) {
            shape = SurfaceCursor::Default;
        } else if (markerAt(cursor.position)) {
            shape = SurfaceCursor::ResizeEW;
        } else if (splitter.hovered(cursor.position, layout, interaction.viewSize,
                                    interaction.scale, enabled)) {
            shape = SurfaceCursor::ResizeNS;
        }
    }
    surfaceSetCursor(shape);
}

void SignalViewImpl::updateState() {
    const F32 maxTranslation = std::abs((1.0f / interaction.zoom) - 1.0f);
    const F32 translation =
        std::clamp(-2.0f * interaction.offset, -maxTranslation, maxTranslation);

    // Update global pixel size.

    pixelSize = {
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y
    };

    // Update axis component (computes paddingScale internally).

    axis->updatePixelSize(pixelSize);
    const auto& paddingScale = axis->paddingScale();

    const bool combined = lineplotEnabled && waterfallEnabled;
    const auto panels = detail::CalculateSignalViewPanels(paddingScale,
                                                           interaction.viewSize,
                                                           splitter.ratio);
    const F32 linePanelScale = combined ? panels.lineFraction : 1.0f;
    axis->updateVerticalScale(linePanelScale);

    // Update the lineplot layer.

    if (lineplotEnabled) {
        auto signalTransform = glm::mat4(1.0f);

        const F32 linePanelOffset = paddingScale.y * (1.0f - linePanelScale);
        signalTransform = glm::translate(signalTransform,
                                         glm::vec3(translation *
                                                       paddingScale.x *
                                                       interaction.zoom,
                                                   linePanelOffset, 0.0f));
        signalTransform = glm::scale(signalTransform,
                                     glm::vec3(paddingScale.x,
                                               paddingScale.y * linePanelScale,
                                               1.0f));

        signalUniforms.transform = signalTransform;
        signalUniforms.thickness[0] = pixelSize.x * kLineThickness * 3.0f;
        signalUniforms.thickness[1] =
            pixelSize.y * kLineThickness * 3.0f / linePanelScale;
        signalUniforms.zoom = interaction.zoom;
        signalUniforms.numberOfPoints = numberOfElements;

        holdUniforms.transform = signalTransform;
        holdUniforms.thickness[0] = pixelSize.x * kLineThickness * 3.0f;
        holdUniforms.thickness[1] =
            pixelSize.y * kLineThickness * 3.0f / linePanelScale;
        holdUniforms.zoom = interaction.zoom;
        holdUniforms.numberOfPoints = numberOfElements;
    }

    // Clip signal and cursor to the plot area.

    const auto& vs = interaction.viewSize;
    const auto& plotRect = panels.plot;
    if (combined) {
        const auto& lineRect = panels.line;
        waterfallProgram->scissorRect(panels.waterfall);

        waterfallUniforms.panelScaleX = paddingScale.x;
        waterfallUniforms.panelScaleY = paddingScale.y * (1.0f - linePanelScale);
        waterfallUniforms.panelOffsetY = -paddingScale.y * linePanelScale;
        signalProgram->scissorRect(lineRect);
        if (fill) {
            fillProgram->scissorRect(lineRect);
        }
        if (maxHold) {
            maxHoldProgram->scissorRect(lineRect);
        }
    } else if (lineplotEnabled) {
        signalProgram->scissorRect(plotRect);
        if (fill) {
            fillProgram->scissorRect(plotRect);
        }
        if (maxHold) {
            maxHoldProgram->scissorRect(plotRect);
        }
    } else if (waterfallEnabled) {
        waterfallProgram->scissorRect(plotRect);
        waterfallUniforms.panelScaleX = paddingScale.x;
        waterfallUniforms.panelScaleY = paddingScale.y;
        waterfallUniforms.panelOffsetY = 0.0f;
    }
    axis->updateScissorRect({0, 0,
                             static_cast<U32>(vs.x),
                             static_cast<U32>(vs.y)});

    // Update the labels.

    updateLabelState();

    // Schedule the uniform buffers for update.

    updateSignalUniformBufferFlag = true;
}

void SignalViewImpl::updateLabelState() {
    const auto& paddingScale = axis->paddingScale();
    const bool combined = lineplotEnabled && waterfallEnabled;
    const bool hasFreqAttrs =
        input.hasAttribute("frequency") &&
        input.hasAttribute("sampleRate");
    const F32 centerFreq =
        hasFreqAttrs ? std::any_cast<F32>(input.attribute("frequency")) : 0.0f;
    const F32 sampleRate =
        hasFreqAttrs ? std::any_cast<F32>(input.attribute("sampleRate")) : 0.0f;

    const F32 translation =
        std::clamp(-2.0f * interaction.offset,
                   -std::abs((1.0f / interaction.zoom) - 1.0f),
                   std::abs((1.0f / interaction.zoom) - 1.0f));

    // Update tick labels via axis component.

    if (axis) {
        const bool ticksVisible = lineplotEnabled &&
            interaction.placement != SurfacePlacementType::Attached;
        axis->setShowFrameTicks(ticksVisible);

        auto xFormatter = [hasFreqAttrs, centerFreq, sampleRate,
                           hasLineplot = lineplotEnabled,
                           zoom = interaction.zoom, translation](const F32 position) {
            const F32 normalizedPos = position / zoom - translation;
            if (hasFreqAttrs) {
                const F32 freq =
                    (centerFreq + normalizedPos * sampleRate / 2.0f) / 1e6f;
                return jst::fmt::format("{:.02f}", freq);
            }
            const F32 value = hasLineplot
                ? normalizedPos
                : (normalizedPos + 1.0f) * 0.5f;
            return jst::fmt::format("{:.02f}", value);
        };

        Render::Components::Axis::TickFormatter yFormatter;
        if (lineplotEnabled) {
            yFormatter = [min = rangeMin, max = rangeMax](const F32 position) {
                return detail::LineplotAmplitudeLabel(position, min, max);
            };
        }

        axis->updateTickFormatters(std::move(xFormatter), std::move(yFormatter));
    }

    if (text) {
        text->updatePixelSize(pixelSize);
    }
    if (cursorText) {
        cursorText->updatePixelSize(pixelSize);
    }
    if (markerText) {
        markerText->updatePixelSize(pixelSize);
    }
    if (markerBadgeText) {
        markerBadgeText->updatePixelSize(pixelSize);
    }
    if (markerTagText) {
        markerTagText->updatePixelSize(pixelSize);
    }

    if (lineplotEnabled && text) {
        const F32 tickOffset = axis->getConfig().majorTickLengthPx + 4.0f;

        auto holdLabel = text->get("hold");
        const F32 lineHeight = text->getConfig().font
            ? static_cast<F32>(text->getConfig().font->lineHeight()) * holdLabel.scale
            : 0.0f;
        holdLabel.position = {-paddingScale.x + pixelSize.x * tickOffset,
                              paddingScale.y - pixelSize.y * (tickOffset + lineHeight)};
        holdLabel.fill = displayHeld ? "HOLD" : " ";
        text->update("hold", holdLabel);

        auto header = text->get("header");
        if (interaction.placement == SurfacePlacementType::Attached) {
            header.fill = " ";
        } else {
            header.position = {-paddingScale.x + pixelSize.x * tickOffset,
                               paddingScale.y - pixelSize.y * tickOffset};
            if (hasFreqAttrs) {
                header.fill = jst::fmt::format("CENTER {:.3f} MHz   SPAN {:.3f} MHz",
                                               centerFreq / 1e6f, sampleRate / 1e6f);
            } else {
                header.fill = "CENTER 0.000   SPAN 1.000";
            }
        }
        text->update("header", header);

        auto zoomLabel = text->get("zoom");
        zoomLabel.position = {
            paddingScale.x -
                pixelSize.x * (axis->getConfig().majorTickLengthPx + 4.0f),
            paddingScale.y -
                pixelSize.y * (axis->getConfig().majorTickLengthPx + 4.0f),
        };
        zoomLabel.alignment = {2, 0};
        if (splitter.dragging) {
            zoomLabel.fill = jst::fmt::format("SPLIT {:.0f}%", splitter.ratio * 100.0f);
        } else if (interaction.placement == SurfacePlacementType::Attached) {
            zoomLabel.fill = " ";
        } else if (std::abs(interaction.zoom - 1.0f) > 0.01f) {
            zoomLabel.fill = jst::fmt::format("ZOOM {:.1f}x", interaction.zoom);
        } else {
            zoomLabel.fill = " ";
        }
        text->update("zoom", zoomLabel);

        auto amplitudeTitle = text->get("amplitude-title");
        const F32 lineFraction = axis->getConfig().verticalScale;
        amplitudeTitle.position = {-1.0f + pixelSize.x * 3.0f,
                                   paddingScale.y * (1.0f - lineFraction)};
        amplitudeTitle.fill = combined ? amplitudeLabel : " ";
        text->update("amplitude-title", amplitudeTitle);

        auto waterfallTitle = text->get("waterfall-title");
        waterfallTitle.position = {-1.0f + pixelSize.x * 3.0f,
                                   -paddingScale.y * lineFraction};
        waterfallTitle.fill = combined ? waterfallLabel : " ";
        text->update("waterfall-title", waterfallTitle);
    }
}

F32 SignalViewImpl::viewTranslation() const {
    const F32 maxTranslation = std::abs((1.0f / interaction.zoom) - 1.0f);
    return std::clamp(-2.0f * interaction.offset, -maxTranslation, maxTranslation);
}

std::optional<F32> SignalViewImpl::cursorPoint() const {
    const bool visible = cursor.inside &&
                         !splitter.dragging &&
                         interaction.placement != SurfacePlacementType::Attached &&
                         insidePlot(cursor.position) &&
                         numberOfElements >= 2;
    if (!visible) {
        return std::nullopt;
    }
    return pointAtX(cursor.position.x);
}

F32 SignalViewImpl::projectPointX(const F32 xPoint) const {
    return (xPoint + viewTranslation()) * interaction.zoom * axis->paddingScale().x;
}

std::optional<F32> SignalViewImpl::displayedAmplitude(const F32 xPoint) const {
    if (!lineplotEnabled || numberOfElements < 2 ||
        displayedPoints.size() < numberOfElements * 2) {
        return std::nullopt;
    }
    const F32 sample = (xPoint + 1.0f) * 0.5f * (numberOfElements - 1);
    const U64 lower = std::min(static_cast<U64>(std::max(sample, 0.0f)), numberOfElements - 2);
    const F32 fraction = std::clamp(sample - static_cast<F32>(lower), 0.0f, 1.0f);
    const F32 yLower = displayedPoints[(lower * 2) + 1];
    const F32 yUpper = displayedPoints[(lower * 2) + 3];
    return yLower + (yUpper - yLower) * fraction;
}

F32 SignalViewImpl::amplitudeToNdc(const F32 yPoint) const {
    const auto& padding = axis->paddingScale();
    const bool combined = lineplotEnabled && waterfallEnabled;
    const F32 lineFraction = combined ? axis->getConfig().verticalScale : 1.0f;
    return padding.y * (1.0f - lineFraction) +
           padding.y * lineFraction * std::clamp(yPoint, -1.0f, 1.0f);
}

std::string SignalViewImpl::formatPointX(const F32 xPoint) {
    const bool hasFreqAttrs = input.hasAttribute("frequency") &&
                              input.hasAttribute("sampleRate");
    if (hasFreqAttrs) {
        const F32 centerFreq = std::any_cast<F32>(input.attribute("frequency"));
        const F32 sampleRate = std::any_cast<F32>(input.attribute("sampleRate"));
        return jst::fmt::format("{:.4f} MHz", (centerFreq + xPoint * sampleRate / 2.0f) / 1e6f);
    }
    return jst::fmt::format("{:.4f}", lineplotEnabled ? xPoint : (xPoint + 1.0f) * 0.5f);
}

std::string SignalViewImpl::formatSpanX(const F32 delta) {
    const bool hasFreqAttrs = input.hasAttribute("frequency") &&
                              input.hasAttribute("sampleRate");
    if (hasFreqAttrs) {
        const F32 sampleRate = std::any_cast<F32>(input.attribute("sampleRate"));
        return detail::FormatFrequencySpan(std::abs(delta) * sampleRate / 2.0f);
    }
    return jst::fmt::format("{:.4f}", std::abs(lineplotEnabled ? delta : delta * 0.5f));
}

std::string SignalViewImpl::formatAmplitude(const F32 yPoint) const {
    const auto value = detail::LineplotAmplitudeValue(yPoint, rangeMin, rangeMax);
    if (!value) {
        return {};
    }
    const auto unit = detail::LabelUnit(amplitudeLabel);
    return unit.empty()
        ? jst::fmt::format("{:.1f}", *value)
        : jst::fmt::format("{:.1f} {}", *value, unit);
}

void SignalViewImpl::syncMarkers() {
    if (markerDrag.index || configChangePending()) {
        return;
    }
    if (updateMarkersFlag || configChangeEnabled("markers")) {
        markerPositions = markers;
    }
    if (updateMarkersFlag || configChangeEnabled("pins")) {
        applyPins();
    }
    updateMarkersFlag = false;
}

void SignalViewImpl::applyPins() {
    pinned.fill(false);
    for (const auto index : pins) {
        if (index < markerPositions.size() && index < pinned.size()) {
            pinned[index] = true;
        }
    }
}

std::vector<U64> SignalViewImpl::pinnedIndices() const {
    std::vector<U64> indices;
    for (U64 i = 0; i < markerPositions.size() && i < pinned.size(); ++i) {
        if (pinned[i]) {
            indices.push_back(i);
        }
    }
    return indices;
}

bool SignalViewImpl::insidePlot(const Extent2D<F32>& position) const {
    const auto& padding = axis->paddingScale();
    const F32 u = (position.x - 0.5f) / std::max(padding.x, 1e-6f) + 0.5f;
    const F32 v = (position.y - 0.5f) / std::max(padding.y, 1e-6f) + 0.5f;
    return u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f;
}

F32 SignalViewImpl::pointAtX(const F32 x) const {
    const F32 u = (x - 0.5f) / std::max(axis->paddingScale().x, 1e-6f) + 0.5f;
    return std::clamp((u * 2.0f - 1.0f) / interaction.zoom - viewTranslation(), -1.0f, 1.0f);
}

std::optional<U64> SignalViewImpl::markerAt(const Extent2D<F32>& position) const {
    if (interaction.placement == SurfacePlacementType::Attached ||
        numberOfElements < 2 || !insidePlot(position)) {
        return std::nullopt;
    }
    const F32 pointerNdc = position.x * 2.0f - 1.0f;
    F32 best = kMarkerPickRadiusPx * (2.0f * interaction.scale) /
               static_cast<F32>(std::max<U64>(interaction.viewSize.x, 1));
    std::optional<U64> nearest;
    for (U64 i = 0; i < markerPositions.size(); ++i) {
        const F32 distance = std::abs(projectPointX(markerPositions[i]) - pointerNdc);
        if (distance < best) {
            best = distance;
            nearest = i;
        }
    }
    return nearest;
}

std::optional<U64> SignalViewImpl::tagAt(const Extent2D<F32>& position) const {
    const Extent2D<F32> pointer = {position.x * 2.0f - 1.0f, 1.0f - position.y * 2.0f};
    for (U64 i = 0; i < tagBounds.size(); ++i) {
        const auto& bounds = tagBounds[i];
        if (bounds.active &&
            std::abs(pointer.x - bounds.center.x) <= bounds.halfSize.x &&
            std::abs(pointer.y - bounds.center.y) <= bounds.halfSize.y) {
            return i;
        }
    }
    return std::nullopt;
}

void SignalViewImpl::toggleMarker() {
    const auto point = cursorPoint();
    if (!point) {
        return;
    }

    if (const auto nearest = markerAt(cursor.position)) {
        const U64 removed = *nearest;
        markerPositions.erase(markerPositions.begin() + removed);
        std::copy(pinned.begin() + removed + 1, pinned.end(), pinned.begin() + removed);
        pinned.back() = false;
    } else if (markerPositions.size() < detail::MaxMarkers) {
        markerPositions.push_back(*point);
    } else {
        return;
    }
    commitMarkers();
}

void SignalViewImpl::clearMarkers() {
    if (markerPositions.empty()) {
        return;
    }
    markerPositions.clear();
    pinned.fill(false);
    commitMarkers();
}

void SignalViewImpl::commitMarkers() {
    if (!configChangeEnabled("markers")) {
        return;
    }
    Parser::Map edit;
    edit["markers"] = markerPositions;
    if (configChangeEnabled("pins")) {
        edit["pins"] = pinnedIndices();
    }
    if (requestConfigChange(edit) != Result::SUCCESS) {
        markerPositions = markers;
        applyPins();
    }
}

Result SignalViewImpl::updateCursorState() {
    const auto& padding = axis->paddingScale();
    const auto point = cursorPoint();
    const bool visible = point.has_value();

    std::string xLabelText;
    std::string yLabelText;
    F32 xNdc = 0.0f;
    F32 yNdc = 0.0f;
    bool hasMarker = false;

    if (visible) {
        cursor.point = *point;
        xNdc = std::clamp((*point + viewTranslation()) * interaction.zoom,
                          -1.0f, 1.0f) * padding.x;
        xLabelText = formatPointX(*point);
        if (const auto yPoint = displayedAmplitude(*point)) {
            yLabelText = formatAmplitude(*yPoint);
            if (!yLabelText.empty()) {
                yNdc = amplitudeToNdc(*yPoint);
                hasMarker = true;
            }
        }
    }

    cursor.visible = visible;
    cursor.marker = hasMarker;
    cursor.plot = {xNdc, yNdc};

    Extent2D<F32> xLabelPosition = {-2.0f, -2.0f};
    Extent2D<F32> yLabelPosition = {-2.0f, -2.0f};

    if (cursorShapes) {
        JST_CHECK(cursorShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(cursorShapes->getPositions("cursor", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(cursorShapes->getSizes("cursor", sizes));
        for (U64 i = 0; i < kCursorInstances; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }

        if (visible) {
            const F32 scale = interaction.scale;
            const F32 toPixelsX = static_cast<F32>(interaction.viewSize.x) * 0.5f;
            const F32 toPixelsY = static_cast<F32>(interaction.viewSize.y) * 0.5f;

            if (!cursor.overMarker) {
                positions[kCursorLine] = {xNdc, 0.0f};
                sizes[kCursorLine] = {2.0f * scale, padding.y * 2.0f * toPixelsY};
            }

            if (hasMarker && !cursor.overMarker) {
                positions[kCursorHalo] = {xNdc, yNdc};
                sizes[kCursorHalo] = {16.0f * scale, 16.0f * scale};
                positions[kCursorMarker] = {xNdc, yNdc};
                sizes[kCursorMarker] = {11.0f * scale, 11.0f * scale};
            }

            if (cursorText && !cursor.overMarker) {
                const auto& font = cursorText->getConfig().font;
                const F32 lineHeight = font ? font->lineHeight() * kLabelScale : 0.0f;
                const F32 xWidth = cursorText->advance(xLabelText) * kLabelScale;
                const F32 yWidth = yLabelText.empty()
                    ? 0.0f
                    : cursorText->advance(yLabelText) * kLabelScale;
                const F32 gap = yLabelText.empty() ? 0.0f : 10.0f;
                const F32 padX = 9.0f;
                const F32 padY = 4.0f;
                const F32 pillWidth = (padX * 2.0f + xWidth + gap + yWidth) * pixelSize.x;
                const F32 pillHeight = (padY * 2.0f + lineHeight) * pixelSize.y;
                const F32 nudge = 14.0f * pixelSize.x;

                F32 left = xNdc + nudge;
                if (left + pillWidth > padding.x) {
                    left = xNdc - nudge - pillWidth;
                }
                const F32 headerOffset = axis->getConfig().majorTickLengthPx + 4.0f;
                const F32 top = padding.y - (headerOffset + lineHeight + 8.0f) * pixelSize.y;
                const F32 centerY = top - pillHeight * 0.5f;
                const F32 centerX = left + pillWidth * 0.5f;

                positions[kCursorPillEdge] = {centerX, centerY};
                sizes[kCursorPillEdge] = {pillWidth * toPixelsX + 2.0f * scale,
                                          pillHeight * toPixelsY + 2.0f * scale};
                positions[kCursorPill] = {centerX, centerY};
                sizes[kCursorPill] = {pillWidth * toPixelsX, pillHeight * toPixelsY};

                xLabelPosition = {left + padX * pixelSize.x, centerY};
                yLabelPosition = {left + (padX + xWidth + gap) * pixelSize.x, centerY};
            }
        }

        JST_CHECK(cursorShapes->updatePositions("cursor"));
        JST_CHECK(cursorShapes->updateSizes("cursor"));
    }

    if (cursorText) {
        const bool readout = visible && !cursor.overMarker;
        auto xLabelElement = cursorText->get("cursor-x");
        xLabelElement.position = xLabelPosition;
        xLabelElement.fill = readout ? xLabelText : " ";
        JST_CHECK(cursorText->update("cursor-x", xLabelElement));

        auto yLabelElement = cursorText->get("cursor-y");
        yLabelElement.position = yLabelPosition;
        yLabelElement.fill = (readout && !yLabelText.empty()) ? yLabelText : " ";
        JST_CHECK(cursorText->update("cursor-y", yLabelElement));
    }

    return Result::SUCCESS;
}

Result SignalViewImpl::updateMarkerState() {
    const auto& padding = axis->paddingScale();
    const bool shown = numberOfElements >= 2;
    const bool table = interaction.placement != SurfacePlacementType::Attached;

    struct Row {
        std::string tag;
        std::string id;
        std::string x;
        std::string y;
        F32 xNdc = 0.0f;
        F32 yNdc = 0.0f;
        bool inView = false;
        bool dot = false;
        bool tagged = false;
        F32 tagWidth = 0.0f;
        Extent2D<F32> tagPosition = {-2.0f, -2.0f};
        Extent2D<F32> idPosition = {-2.0f, -2.0f};
        Extent2D<F32> xPosition = {-2.0f, -2.0f};
        Extent2D<F32> yPosition = {-2.0f, -2.0f};
    };
    struct Span {
        bool active = false;
        F32 left = 0.0f;
        F32 right = 0.0f;
        std::string label;
        Extent2D<F32> labelPosition = {-2.0f, -2.0f};
    };
    std::array<Row, detail::MaxMarkers> rows;
    std::array<Span, detail::MarkerSpans> spans;
    const U64 count = shown ? std::min<U64>(markerPositions.size(), detail::MaxMarkers) : 0;
    Render::Components::Text* const badgeText =
        markerBadgeText ? markerBadgeText.get() : markerText.get();
    Render::Components::Text* const tagText =
        markerTagText ? markerTagText.get() : badgeText;

    for (U64 i = 0; i < count; ++i) {
        auto& row = rows[i];
        const F32 marker = markerPositions[i];
        row.xNdc = projectPointX(marker);
        row.inView = std::abs(row.xNdc) <= padding.x;
        row.id = jst::fmt::format("M{}", i + 1);
        row.tag = row.id;
        row.x = formatPointX(marker);
        if (const auto yPoint = displayedAmplitude(marker)) {
            row.y = formatAmplitude(*yPoint);
            if (!row.y.empty()) {
                row.yNdc = amplitudeToNdc(*yPoint);
                row.dot = row.inView;
            }
        }
    }

    const F32 scale = interaction.scale;
    const F32 toPixelsX = static_cast<F32>(interaction.viewSize.x) * 0.5f;
    const F32 toPixelsY = static_cast<F32>(interaction.viewSize.y) * 0.5f;
    const std::shared_ptr<Render::Components::Font> font =
        markerText ? markerText->getConfig().font : nullptr;
    const F32 lineHeight = font ? font->lineHeight() * kLabelScale : 0.0f;
    const std::shared_ptr<Render::Components::Font> tagFont =
        tagText ? tagText->getConfig().font : nullptr;
    const F32 tagLineHeight = tagFont ? tagFont->lineHeight() * kLabelScale : 0.0f;
    const F32 headerOffset = axis->getConfig().majorTickLengthPx + 4.0f;
    const F32 headerHeight = table ? tagLineHeight + 8.0f : 0.0f;
    const F32 tagPadX = 6.0f;
    const F32 tagPadY = 2.0f;
    const F32 tagHeight = (tagPadY * 2.0f + tagLineHeight) * pixelSize.y;
    const F32 cursorRowTop = padding.y - (headerOffset + headerHeight) * pixelSize.y;
    const F32 cursorRowCenter = cursorRowTop - (8.0f + tagLineHeight) * pixelSize.y * 0.5f;

    tagBounds.fill({});
    for (U64 i = 0; i < count; ++i) {
        auto& row = rows[i];
        if (!row.inView || !markerText) {
            continue;
        }
        row.tagWidth = (tagPadX * 2.0f + tagText->advance(row.tag) * kLabelScale) * pixelSize.x;
        // Center oversized tags instead of clamping with inverted bounds.
        const F32 tagLimit = std::max(0.0f, padding.x - row.tagWidth * 0.5f);
        row.tagPosition = {
            std::clamp(row.xNdc, -tagLimit, tagLimit),
            cursorRowCenter,
        };
        row.tagged = true;
        tagBounds[i] = {
            .active = true,
            .center = row.tagPosition,
            .halfSize = {row.tagWidth * 0.5f, tagHeight * 0.5f},
        };
    }

    std::optional<U64> hovered;
    if (cursor.inside && !splitter.dragging && shown) {
        hovered = tagAt(cursor.position);
    }
    cursor.overMarker = hovered.has_value() || markerDrag.index.has_value() ||
                        (cursor.inside && !splitter.dragging && markerAt(cursor.position));

    std::vector<U64> focused;
    if (hovered) {
        focused.push_back(*hovered);
    }
    for (U64 i = 0; i < count; ++i) {
        if (pinned[i] && rows[i].tagged && hovered != i) {
            focused.push_back(i);
        }
    }

    if (!focused.empty()) {
        std::vector<U64> order;
        for (U64 i = 0; i < count; ++i) {
            if (rows[i].tagged) {
                order.push_back(i);
            }
        }
        std::sort(order.begin(), order.end(), [&](const U64 a, const U64 b) {
            return rows[a].xNdc < rows[b].xNdc;
        });
        std::array<bool, detail::MarkerSpans> gaps{};
        for (const U64 index : focused) {
            const U64 rank = std::find(order.begin(), order.end(), index) - order.begin();
            if (rank > 0) {
                gaps[rank - 1] = true;
            }
            if (rank + 1 < order.size()) {
                gaps[rank] = true;
            }
        }
        const F32 arrowLength = kMarkerSpanArrowArmPx *
                                std::cos(glm::radians(kMarkerSpanArrowAngleDeg)) * pixelSize.x;
        const F32 labelGap = kMarkerSpanLabelGapPx * pixelSize.x;
        const F32 tagGap = kMarkerSpanTagGapPx * pixelSize.x;
        const auto fillSpan = [&](Span& span, const U64 leftIndex, const U64 rightIndex) {
            const auto& leftRow = rows[leftIndex];
            const auto& rightRow = rows[rightIndex];
            span.left = leftRow.tagPosition.x + leftRow.tagWidth * 0.5f + tagGap;
            span.right = rightRow.tagPosition.x - rightRow.tagWidth * 0.5f - tagGap;
            span.label = formatSpanX(markerPositions[rightIndex] - markerPositions[leftIndex]);
            const F32 labelWidth = markerText->advance(span.label) * kLabelScale * pixelSize.x;
            span.active = span.right - span.left >= (arrowLength + labelGap) * 2.0f + labelWidth;
            span.labelPosition = {(span.left + span.right) * 0.5f, cursorRowCenter};
        };
        for (U64 gap = 0; gap < detail::MarkerSpans; ++gap) {
            if (gaps[gap]) {
                fillSpan(spans[gap], order[gap], order[gap + 1]);
            }
        }
    }

    if (markerShapes) {
        JST_CHECK(markerShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(markerShapes->getPositions("markers", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(markerShapes->getSizes("markers", sizes));
        for (U64 i = 0; i < kMarkerGroups * detail::MaxMarkers; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }

        std::span<Extent2D<F32>> tablePositions;
        std::span<Extent2D<F32>> tableSizes;
        if (markerTableShapes) {
            JST_CHECK(markerTableShapes->updatePixelSize({
                2.0f / interaction.viewSize.x,
                2.0f / interaction.viewSize.y,
            }));
            JST_CHECK(markerTableShapes->getPositions("table", tablePositions));
            JST_CHECK(markerTableShapes->getSizes("table", tableSizes));
            for (U64 i = 0; i < kMarkerTableGroups * detail::MaxMarkers; ++i) {
                tablePositions[i] = {-2.0f, -2.0f};
                tableSizes[i] = {0.0f, 0.0f};
            }
        }

        const F32 padX = 9.0f;
        const F32 padY = 4.0f;
        const F32 rowGap = 4.0f;
        const F32 pillHeight = (padY * 2.0f + lineHeight) * pixelSize.y;
        const F32 right = padding.x - kMarkerTableGapPx * pixelSize.x;
        const F32 bottom = -padding.y + kMarkerTableGapPx * pixelSize.y;

        for (U64 i = 0; i < count; ++i) {
            auto& row = rows[i];

            if (row.inView) {
                positions[MarkerInstance(kMarkerLine, i)] = {row.xNdc, 0.0f};
                sizes[MarkerInstance(kMarkerLine, i)] = {2.0f * scale, padding.y * 2.0f * toPixelsY};
            }

            if (row.dot) {
                positions[MarkerInstance(kMarkerHalo, i)] = {row.xNdc, row.yNdc};
                sizes[MarkerInstance(kMarkerHalo, i)] = {13.0f * scale, 13.0f * scale};
                positions[MarkerInstance(kMarkerDot, i)] = {row.xNdc, row.yNdc};
                sizes[MarkerInstance(kMarkerDot, i)] = {8.0f * scale, 8.0f * scale};
            }

            if (markerText && table) {
                const F32 idWidth = badgeText->advance(row.id) * kLabelScale;
                const F32 idGap = 8.0f;
                const F32 xWidth = markerText->advance(row.x) * kLabelScale;
                const F32 yWidth = row.y.empty() ? 0.0f : markerText->advance(row.y) * kLabelScale;
                const F32 gap = row.y.empty() ? 0.0f : 10.0f;
                const F32 pillWidth =
                    (padX * 2.0f + idWidth + idGap + xWidth + gap + yWidth) * pixelSize.x;
                const F32 left = right - pillWidth;
                const F32 centerY = bottom + (pillHeight + rowGap * pixelSize.y) * (count - 1 - i) +
                                    pillHeight * 0.5f;
                const F32 centerX = left + pillWidth * 0.5f;

                if (markerTableShapes) {
                    tablePositions[MarkerInstance(kMarkerPillEdge, i)] = {centerX, centerY};
                    tableSizes[MarkerInstance(kMarkerPillEdge, i)] = {
                        pillWidth * toPixelsX + 2.0f * scale,
                        pillHeight * toPixelsY + 2.0f * scale,
                    };
                    tablePositions[MarkerInstance(kMarkerPill, i)] = {centerX, centerY};
                    tableSizes[MarkerInstance(kMarkerPill, i)] = {pillWidth * toPixelsX,
                                                                  pillHeight * toPixelsY};
                }

                row.idPosition = {left + padX * pixelSize.x, centerY};
                row.xPosition = {left + (padX + idWidth + idGap) * pixelSize.x, centerY};
                row.yPosition = {left + (padX + idWidth + idGap + xWidth + gap) * pixelSize.x,
                                 centerY};
            }
        }

        JST_CHECK(markerShapes->updatePositions("markers"));
        JST_CHECK(markerShapes->updateSizes("markers"));
        if (markerTableShapes) {
            JST_CHECK(markerTableShapes->updatePositions("table"));
            JST_CHECK(markerTableShapes->updateSizes("table"));
        }
    }

    if (markerTagShapes) {
        JST_CHECK(markerTagShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));
        JST_CHECK(markerTagShapes->updateProperties("tags", 4.0f * interaction.scale, 0.0f, {}));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(markerTagShapes->getPositions("tags", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(markerTagShapes->getSizes("tags", sizes));
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }

        for (U64 i = 0; i < count; ++i) {
            const auto& row = rows[i];
            if (!row.tagged) {
                continue;
            }
            positions[i] = row.tagPosition;
            sizes[i] = {row.tagWidth * toPixelsX, tagHeight * toPixelsY};
        }

        JST_CHECK(markerTagShapes->updatePositions("tags"));
        JST_CHECK(markerTagShapes->updateSizes("tags"));
    }

    if (markerSpanShapes) {
        JST_CHECK(markerSpanShapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        }));

        const F32 thickness = kMarkerSpanThicknessPx * scale;
        JST_CHECK(markerSpanShapes->updateProperties("spans", thickness * 0.5f, 0.0f, {}));
        JST_CHECK(markerSpanShapes->updateProperties("arrows", thickness * 0.5f, 0.0f, {}));

        std::span<Extent2D<F32>> positions;
        JST_CHECK(markerSpanShapes->getPositions("spans", positions));
        std::span<Extent2D<F32>> sizes;
        JST_CHECK(markerSpanShapes->getSizes("spans", sizes));
        for (U64 i = 0; i < kSpanSegments * detail::MarkerSpans; ++i) {
            positions[i] = {-2.0f, -2.0f};
            sizes[i] = {0.0f, 0.0f};
        }
        std::span<Extent2D<F32>> arrowPositions;
        JST_CHECK(markerSpanShapes->getPositions("arrows", arrowPositions));
        std::span<Extent2D<F32>> arrowSizes;
        JST_CHECK(markerSpanShapes->getSizes("arrows", arrowSizes));
        std::span<F32> arrowRotations;
        JST_CHECK(markerSpanShapes->getRotations("arrows", arrowRotations));
        for (U64 i = 0; i < kSpanArrows * detail::MarkerSpans; ++i) {
            arrowPositions[i] = {-2.0f, -2.0f};
            arrowSizes[i] = {0.0f, 0.0f};
            arrowRotations[i] = 0.0f;
        }

        for (U64 i = 0; i < detail::MarkerSpans; ++i) {
            const auto& span = spans[i];
            if (!span.active || !markerText) {
                continue;
            }
            const F32 arrowAngle = glm::radians(kMarkerSpanArrowAngleDeg);
            const F32 armReach = (kMarkerSpanArrowArmPx - kMarkerSpanThicknessPx) * 0.5f;
            const F32 armDx = armReach * std::cos(arrowAngle) * pixelSize.x;
            const F32 armDy = armReach * std::sin(arrowAngle) * pixelSize.y;
            const F32 labelGap = kMarkerSpanLabelGapPx * pixelSize.x;
            const F32 labelHalf = markerText->advance(span.label) * kLabelScale * pixelSize.x * 0.5f;
            const F32 leadStart = span.left;
            const F32 trailEnd = span.right;
            const F32 leadEnd = std::max(span.labelPosition.x - labelHalf - labelGap, leadStart);
            const F32 trailStart = std::min(span.labelPosition.x + labelHalf + labelGap, trailEnd);

            const auto arm = [&](const U64 arrow, const F32 tipX, const F32 dx, const F32 dy, const F32 degrees) {
                arrowPositions[ArrowInstance(i, arrow)] = {tipX + dx, cursorRowCenter + dy};
                arrowSizes[ArrowInstance(i, arrow)] = {kMarkerSpanArrowArmPx * scale, thickness};
                arrowRotations[ArrowInstance(i, arrow)] = degrees;
            };
            arm(kSpanLeadUpperArm, span.left, armDx, armDy, kMarkerSpanArrowAngleDeg);
            arm(kSpanLeadLowerArm, span.left, armDx, -armDy, -kMarkerSpanArrowAngleDeg);
            arm(kSpanTrailUpperArm, span.right, -armDx, armDy, -kMarkerSpanArrowAngleDeg);
            arm(kSpanTrailLowerArm, span.right, -armDx, -armDy, kMarkerSpanArrowAngleDeg);

            positions[SpanInstance(i, kSpanLeadSegment)] = {(leadStart + leadEnd) * 0.5f, cursorRowCenter};
            sizes[SpanInstance(i, kSpanLeadSegment)] = {(leadEnd - leadStart) * toPixelsX, thickness};
            positions[SpanInstance(i, kSpanTrailSegment)] = {(trailStart + trailEnd) * 0.5f, cursorRowCenter};
            sizes[SpanInstance(i, kSpanTrailSegment)] = {(trailEnd - trailStart) * toPixelsX, thickness};
        }

        JST_CHECK(markerSpanShapes->updatePositions());
        JST_CHECK(markerSpanShapes->updateSizes());
        JST_CHECK(markerSpanShapes->updateRotations());
    }

    if (markerText) {
        for (U64 i = 0; i < detail::MaxMarkers; ++i) {
            const auto& row = rows[i];
            const bool active = i < count;

            auto tag = tagText->get(MarkerElement(i, "tag"));
            tag.position = row.tagPosition;
            tag.fill = (active && row.inView) ? row.tag : " ";
            JST_CHECK(tagText->update(MarkerElement(i, "tag"), tag));

            auto id = badgeText->get(MarkerElement(i, "id"));
            id.position = row.idPosition;
            id.fill = (active && table) ? row.id : " ";
            JST_CHECK(badgeText->update(MarkerElement(i, "id"), id));

            auto x = markerText->get(MarkerElement(i, "x"));
            x.position = row.xPosition;
            x.fill = (active && table) ? row.x : " ";
            JST_CHECK(markerText->update(MarkerElement(i, "x"), x));

            auto y = markerText->get(MarkerElement(i, "y"));
            y.position = row.yPosition;
            y.fill = (active && table && !row.y.empty()) ? row.y : " ";
            JST_CHECK(markerText->update(MarkerElement(i, "y"), y));
        }

        for (U64 i = 0; i < detail::MarkerSpans; ++i) {
            const auto& span = spans[i];

            auto label = markerText->get(SpanElement(i, "label"));
            label.position = span.labelPosition;
            label.fill = span.active ? span.label : " ";
            JST_CHECK(markerText->update(SpanElement(i, "label"), label));
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
