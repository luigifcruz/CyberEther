#include "module_impl.hh"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <limits>

#include "jetstream/memory/axis.hh"
#include "jetstream/tools/numeric.hh"

namespace Jetstream::Modules {

Result SignalViewImpl::validate() {
    const auto& config = *candidate();
    const bool hasLineplot = detail::SignalViewHasLineplot(config.mode);
    const bool hasWaterfall = detail::SignalViewHasWaterfall(config.mode);
    const bool hasWaterfall3D = detail::SignalViewHasWaterfall3D(config.mode);

    validatedNumberOfElements = 0;
    validatedNumberOfBatches = 0;
    validatedInputElementStride = 0;
    validatedInputBatchStride = 0;
    validatedNormalizationFactor = 0.0f;
    validatedLineplotEnabled = false;
    validatedWaterfallEnabled = false;
    validatedWaterfall3dEnabled = false;

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

    if (!Render::Colormap::Valid(config.colormap)) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Invalid colormap '{}'.", config.colormap);
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

    if (!std::isfinite(config.splitRatio) ||
        config.splitRatio < detail::MinSplitRatio || config.splitRatio > detail::MaxSplitRatio) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Split ratio must be between 0.1 and 0.9.");
        return Result::ERROR;
    }

    if (hasWaterfall &&
        (config.waterfallHeight == 0 || config.waterfallHeight > 8192)) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Invalid waterfall height value '{}', "
                  "must be between 1 and 8192.",
                  config.waterfallHeight);
        return Result::ERROR;
    }

    if (hasWaterfall3D && config.waterfallHeight < 2) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] 3D waterfall needs at least 2 rows.");
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
    if ((hasLineplot || hasWaterfall3D) && numberOfElements < 2) {
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

    if (hasWaterfall3D) {
        U64 slotCount = 0;
        if (!Jetstream::detail::CheckedMultiply(numberOfElements - 1, 6, slotCount) ||
            slotCount > static_cast<U64>(std::numeric_limits<I32>::max())) {
            JST_ERROR("[MODULE_SIGNAL_VIEW] 3D waterfall geometry exceeds the "
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
    validatedWaterfall3dEnabled = hasWaterfall3D;

    return Result::SUCCESS;
}

Result SignalViewImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("signal"));

    return Result::SUCCESS;
}

Result SignalViewImpl::create() {
    canvas.reset(*this);
    updateLayoutFlag = false;

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
    waterfall3dEnabled = validatedWaterfall3dEnabled;
    waterfallAveragingCount = 0;
    maxHoldWarmupBlocks = 0;
    lineplotAveragingInitialized = false;
    waterfallHistory = {};
    updateSignalPointsFlag = false;
    updateHoldPointsFlag = false;

    lineplot.configure({
        .numberOfElements = numberOfElements,
        .fill = fill,
        .maxHold = maxHold,
    });
    waterfall.configure({
        .width = numberOfElements,
        .height = waterfallHeight,
    });

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
        canvas.updateMarkersFlag |= config.markers != markers || config.pins != pins;
        markers = config.markers;
        pins = config.pins;
        lineplotAveraging = config.lineplotAveraging;
        waterfallAveraging = config.waterfallAveraging;
        rangeMin = config.rangeMin;
        rangeMax = config.rangeMax;
        colormap = config.colormap;
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

    if (!window->hasFont("default_mono")) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Font 'default_mono' not found.");
        return Result::ERROR;
    }

    if (waterfall3dEnabled) {
        JST_CHECK(waterfall3d.create(window, numberOfElements, waterfallHeight, colormap));
        waterfallHistory.dirtyRows = waterfallHeight;
        JST_CHECK(surfaceCreateManifest({
            .id = "default",
            .size = waterfall3d.viewSize(),
            .surface = waterfall3d.framebuffer(),
        }));
        return Result::SUCCESS;
    }

    if (lineplotEnabled) {
        JST_CHECK(lineplot.create(window, {
            .signalPoints = signalPoints,
            .signalVertices = signalVertices,
            .fillVertices = fillVertices,
            .maxHoldPoints = maxHoldPoints,
            .maxHoldVertices = maxHoldVertices,
        }));
    }

    if (waterfallEnabled) {
        JST_CHECK(waterfall.create(window, waterfallBins, colormap));
    }

    JST_CHECK(canvas.create(window, canvasContext()));

    JST_CHECK(surfaceCreateManifest({
        .id = "default",
        .size = canvas.interaction.viewSize,
        .surface = canvas.framebufferTexture,
    }));

    return Result::SUCCESS;
}

Result SignalViewImpl::destroyPresent() {
    auto& window = render();

    if (!window) {
        return Result::SUCCESS;
    }

    if (waterfall3dEnabled) {
        return waterfall3d.destroy(window);
    }

    return canvas.destroy(window);
}

Result SignalViewImpl::present() {
    if (waterfall3dEnabled) {
        bool viewChanged = false;
        JST_CHECK(waterfall3d.present(surfaceConsumeSurfaceEvents(),
                                      surfaceConsumeInputEvents(),
                                      waterfallFrame(), waterfall3dLabels(), colormap,
                                      viewChanged));
        if (!waterfall3d.held()) {
            waterfallHistory.clearDirty();
        }
        if (viewChanged) {
            surfaceUpdateManifestSize("default", waterfall3d.viewSize());
        }
        return Result::SUCCESS;
    }

    if (!canvas.renderSurface) {
        return Result::SUCCESS;
    }

    const auto context = canvasContext();

    JST_CHECK(canvas.processSurfaceEvents(surfaceConsumeSurfaceEvents()));
    surfaceSetCursor(canvas.processInputEvents(surfaceConsumeInputEvents(), context, {
        .splitEnabled = lineplotEnabled && waterfallEnabled && configChangeEnabled("splitRatio"),
        .pending = [this] { return configChangePending(); },
        .enabled = [this](const std::string& name) { return configChangeEnabled(name); },
        .request = [this](const Parser::Map& edit) { return requestConfigChange(edit); },
    }));

    if (canvas.interaction.viewChanged || updateLayoutFlag) {
        canvas.resize();
        surfaceUpdateManifestSize("default", canvas.interaction.viewSize);
        canvas.updateState(context);
        updateLayoutFlag = false;
    }

    if (waterfallEnabled) {
        if (!canvas.displayHeld) {
            JST_CHECK(waterfall.update(waterfallFrame()));
            waterfallHistory.clearDirty();
        }
        JST_CHECK(waterfall.present(canvas.interaction, colormap));
    }

    if (lineplotEnabled) {
        if (updateSignalPointsFlag && !canvas.displayHeld) {
            JST_CHECK(lineplot.upload(signalPoints, updateHoldPointsFlag));
            updateSignalPointsFlag = false;
        }
        JST_CHECK(lineplot.present());
    }

    return canvas.present(context);
}

SignalViewCanvas::Context SignalViewImpl::canvasContext() {
    return {
        .config = *this,
        .frequency = SignalViewFrequencyOf(input),
        .numberOfElements = numberOfElements,
        .lineplot = lineplotEnabled ? &lineplot : nullptr,
        .waterfall = waterfallEnabled ? &waterfall : nullptr,
    };
}

WaterfallFrame SignalViewImpl::waterfallFrame() const {
    return {
        .bins = waterfallBins.data<F32>(),
        .writeIndex = waterfallHistory.writeIndex,
        .dirty = waterfallHistory.dirtyPlan(waterfallHeight),
    };
}

SignalViewWaterfall3DLabels SignalViewImpl::waterfall3dLabels() const {
    const auto frequency = SignalViewFrequencyOf(input);
    SignalViewWaterfall3DLabels labels;
    labels.frequency = xLabel;
    labels.time = waterfallLabel;
    labels.amplitude = amplitudeLabel;
    labels.hasFrequency = frequency.valid;
    labels.centerFrequency = frequency.center;
    labels.sampleRate = frequency.sampleRate;
    labels.rangeMin = rangeMin;
    labels.rangeMax = rangeMax;
    return labels;
}

}  // namespace Jetstream::Modules
