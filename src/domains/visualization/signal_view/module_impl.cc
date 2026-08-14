#include "module_impl.hh"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

#include "jetstream/constants.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/tools/numeric.hh"
#include "resources/shaders/global_shaders.hh"
#include "resources/shaders/signal_view_shaders.hh"

namespace Jetstream::Modules {

namespace {

constexpr F32 kLineThickness = 1.0f;

}  // namespace

Result SignalViewImpl::validate() {
    const auto& config = *candidate();
    const bool hasLineplot = detail::SignalViewHasLineplot(config.mode);
    const bool hasWaterfall = detail::SignalViewHasWaterfall(config.mode);

    validatedNumberOfElements = 0;
    validatedNumberOfBatches = 0;
    validatedInputElementCount = 0;
    validatedInputElementStride = 0;
    validatedInputBatchStride = 0;
    validatedNormalizationFactor = 0.0f;
    validatedLineplotEnabled = false;
    validatedWaterfallEnabled = false;

    if (!hasLineplot && !hasWaterfall) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (hasLineplot && config.decimation == 0) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Decimation must be at least 1.");
        return Result::ERROR;
    }

    if (hasLineplot && config.averaging == 0) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Averaging must be at least 1.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.rangeMin) || !std::isfinite(config.rangeMax) ||
        (hasLineplot && config.rangeMin >= config.rangeMax)) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Display range must be finite and ordered.");
        return Result::ERROR;
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

    const U64 inputElementCount = inputTensor.shape(*elementAxis);
    const U64 numberOfElements = hasLineplot
        ? inputElementCount / config.decimation
        : inputElementCount;
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
        if (!Jetstream::detail::CheckedMultiply(inputElementCount,
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
    validatedInputElementCount = inputElementCount;
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
    inputElementCount = validatedInputElementCount;
    inputElementStride = validatedInputElementStride;
    inputBatchStride = validatedInputBatchStride;
    normalizationFactor = validatedNormalizationFactor;
    lineplotEnabled = validatedLineplotEnabled;
    waterfallEnabled = validatedWaterfallEnabled;
    maxHoldWarmupBlocks = 0;
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
                                       {waterfallHeight, inputElementCount},
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
        config.decimation == decimation &&
        config.maxHold == maxHold &&
        config.fill == fill &&
        config.waterfallHeight == waterfallHeight &&
        config.xLabel == xLabel &&
        config.amplitudeLabel == amplitudeLabel &&
        config.waterfallLabel == waterfallLabel) {
        const bool averagingChanged = config.averaging != averaging;
        const bool rangeChanged =
            config.rangeMin != rangeMin || config.rangeMax != rangeMax;
        averaging = config.averaging;
        rangeMin = config.rangeMin;
        rangeMax = config.rangeMax;
        if (averagingChanged) {
            JST_CHECK(resetLineplotHistory());
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

    return resetAveragingState();
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
        cfg.verticalScale = combined ? 0.5f : 1.0f;
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

    // Text labels (header + zoom readouts).

    if (lineplotEnabled) {
        {
            Render::Components::Text::Config cfg;
            cfg.maxCharacters = 256;
            cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
            cfg.font = window->font("default_mono");
            cfg.elements = {
                {"header",
                 {.scale = 0.85f,
                  .position = {-1.0f, 1.0f},
                  .alignment = {0, 0}}},
                {"zoom",
                 {.scale = 0.85f,
                  .position = {0.0f, 1.0f},
                  .alignment = {1, 0}}},
                {"amplitude-title",
                 {.scale = 0.85f,
                  .position = {-1.0f, 0.5f},
                  .alignment = {1, 0},
                  .rotationDeg = 90.0f}},
                {"waterfall-title",
                 {.scale = 0.85f,
                  .position = {-1.0f, -0.5f},
                  .alignment = {1, 0},
                  .rotationDeg = 90.0f}},
            };
            JST_CHECK(window->build(text, cfg));
            JST_CHECK(window->bind(text));
        }
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
        JST_CHECK(axis->surfaceOverlay(cfg));
        if (lineplotEnabled) {
            JST_CHECK(text->surface(cfg));
        }
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
                                            surfaceConsumeSurfaceEvents(),
                                            surfaceConsumeMouseEvents());

    if (interaction.viewChanged) {
        renderSurface->size(interaction.viewSize);
        renderSurface->clearColor(interaction.backgroundColor);
        surfaceUpdateManifestSize("default", interaction.viewSize);
        updateState();
    }

    if (waterfallEnabled) {
        const auto dirtyPlan = waterfallHistory.dirtyPlan(waterfallHeight);
        if (dirtyPlan.firstRowCount > 0) {
            JST_CHECK(waterfallBuffer->update(dirtyPlan.startRow *
                                                  inputElementCount,
                                              dirtyPlan.firstRowCount *
                                                  inputElementCount));
        }
        if (dirtyPlan.secondRowCount > 0) {
            JST_CHECK(waterfallBuffer->update(0,
                                              dirtyPlan.secondRowCount *
                                                  inputElementCount));
        }
        waterfallHistory.clearDirty();

        waterfallUniforms.width = static_cast<int>(inputElementCount);
        waterfallUniforms.height = static_cast<int>(waterfallHeight);
        waterfallUniforms.index = waterfallHistory.writeIndex /
                                  static_cast<F32>(waterfallHeight);
        waterfallUniforms.offset = interaction.offset +
            0.5f * (1.0f - 1.0f / interaction.zoom);
        waterfallUniforms.zoom = interaction.zoom;
        JST_CHECK(waterfallUniformBuffer->update());
    }

    // Process update flags.

    if (lineplotEnabled && updateSignalPointsFlag) {
        JST_CHECK(signalPointsBuffer->update());
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
        updateLabelState();
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

    JST_CHECK(axis->present());
    if (lineplotEnabled) {
        JST_CHECK(text->present());
    }

    return Result::SUCCESS;
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

    // Update the lineplot layer.

    if (lineplotEnabled) {
        auto signalTransform = glm::mat4(1.0f);

        const F32 linePanelScale = combined ? 0.5f : 1.0f;
        const F32 linePanelOffset = combined ? paddingScale.y * 0.5f : 0.0f;
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
    Render::ScissorRect plotRect;
    plotRect.x = static_cast<U32>((1.0f - paddingScale.x) / 2.0f * vs.x);
    plotRect.y = static_cast<U32>((1.0f - paddingScale.y) / 2.0f * vs.y);
    plotRect.width = static_cast<U32>(paddingScale.x * vs.x);
    plotRect.height = static_cast<U32>(paddingScale.y * vs.y);
    if (combined) {
        auto lineRect = plotRect;
        lineRect.height = (plotRect.height + 1) / 2;
        Render::ScissorRect waterfallRect = plotRect;
        waterfallRect.y += lineRect.height;
        waterfallRect.height -= lineRect.height;
        waterfallProgram->scissorRect(waterfallRect);

        waterfallUniforms.panelScaleX = paddingScale.x;
        waterfallUniforms.panelScaleY = paddingScale.y * 0.5f;
        waterfallUniforms.panelOffsetY = -paddingScale.y * 0.5f;
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
                const F32 db = min +
                    ((position + 1.0f) * 0.5f) * (max - min);
                return jst::fmt::format("{:.0f}", db);
            };
        }

        axis->updateTickFormatters(std::move(xFormatter), std::move(yFormatter));
    }

    if (lineplotEnabled && text) {
        text->updatePixelSize(pixelSize);

        auto header = text->get("header");
        if (interaction.placement == SurfacePlacementType::Attached) {
            header.fill = " ";
        } else {
            const F32 tickOffset = axis->getConfig().majorTickLengthPx + 4.0f;
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
        if (interaction.placement == SurfacePlacementType::Attached) {
            zoomLabel.fill = " ";
        } else if (std::abs(interaction.zoom - 1.0f) > 0.01f) {
            zoomLabel.fill = jst::fmt::format("ZOOM {:.1f}x", interaction.zoom);
        } else {
            zoomLabel.fill = " ";
        }
        text->update("zoom", zoomLabel);

        auto amplitudeTitle = text->get("amplitude-title");
        amplitudeTitle.position = {-1.0f + pixelSize.x * 3.0f,
                                   paddingScale.y * 0.5f};
        amplitudeTitle.fill = combined ? amplitudeLabel : " ";
        text->update("amplitude-title", amplitudeTitle);

        auto waterfallTitle = text->get("waterfall-title");
        waterfallTitle.position = {-1.0f + pixelSize.x * 3.0f,
                                   -paddingScale.y * 0.5f};
        waterfallTitle.fill = combined ? waterfallLabel : " ";
        text->update("waterfall-title", waterfallTitle);
    }
}

}  // namespace Jetstream::Modules
