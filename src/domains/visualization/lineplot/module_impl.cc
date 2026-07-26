#include "module_impl.hh"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec2.hpp>
#include <limits>

#include "jetstream/constants.hh"
#include "jetstream/tools/numeric.hh"
#include "resources/shaders/global_shaders.hh"
#include "resources/shaders/lineplot_shaders.hh"

namespace Jetstream::Modules {

Result LineplotImpl::validate() {
    const auto& config = *candidate();

    if (config.decimation == 0) {
        JST_ERROR("[MODULE_LINEPLOT] Decimation must be at least 1.");
        return Result::ERROR;
    }

    if (config.averaging == 0) {
        JST_ERROR("[MODULE_LINEPLOT] Averaging must be at least 1.");
        return Result::ERROR;
    }

    if (config.numberOfVerticalLines < 2) {
        JST_ERROR("[MODULE_LINEPLOT] Number of vertical lines must be at least 2.");
        return Result::ERROR;
    }

    if (config.numberOfHorizontalLines < 2) {
        JST_ERROR("[MODULE_LINEPLOT] Number of horizontal lines must be at least 2.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.thickness) || config.thickness <= 0.0f) {
        JST_ERROR("[MODULE_LINEPLOT] Thickness must be finite and positive.");
        return Result::ERROR;
    }

    const U64 maxRenderScalarCount = std::min(
        static_cast<U64>(std::numeric_limits<std::size_t>::max()) / sizeof(F32),
        static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(F32));

    // Preflight Axis's line strips and fixed 128-character label buffers.
    U64 totalGridLines = 0;
    U64 gridPointScalarCount = 0;
    U64 gridVertexScalarCount = 0;
    U64 gridVertexCount = 0;
    U64 textVertexCount = 0;
    const U64 maxTextVertexCount = std::min(
        static_cast<U64>(std::numeric_limits<std::size_t>::max()) /
            sizeof(glm::vec2),
        static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) /
            sizeof(glm::vec2));
    if (!Jetstream::detail::CheckedAdd(config.numberOfVerticalLines,
                                       config.numberOfHorizontalLines,
                                       totalGridLines) ||
        !Jetstream::detail::CheckedMultiply(totalGridLines, 4,
                                            gridPointScalarCount) ||
        !Jetstream::detail::CheckedMultiply(totalGridLines, 24,
                                            gridVertexScalarCount) ||
        !Jetstream::detail::CheckedMultiply(totalGridLines, 6,
                                            gridVertexCount) ||
        !Jetstream::detail::CheckedMultiply(totalGridLines - 2, 128 * 4,
                                            textVertexCount) ||
        totalGridLines > std::numeric_limits<U32>::max() ||
        gridVertexCount > std::numeric_limits<U32>::max() ||
        gridPointScalarCount > maxRenderScalarCount ||
        gridVertexScalarCount > maxRenderScalarCount ||
        textVertexCount > maxTextVertexCount) {
        JST_ERROR("[MODULE_LINEPLOT] Grid geometry exceeds the supported "
                  "rendering range.");
        return Result::ERROR;
    }

    if (!inputs().contains("signal")) {
        validatedNumberOfElements = 0;
        validatedNumberOfBatches = 0;
        validatedInputRowWidth = 0;
        validatedNormalizationFactor = 0.0f;
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape()) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() == 0 || inputTensor.rank() > 2) {
        JST_ERROR("[MODULE_LINEPLOT] Invalid input rank ({}), expected 1 or 2.",
                  inputTensor.rank());
        return Result::ERROR;
    }

    if (inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const U64 candidateInputRowWidth = inputTensor.shape(inputTensor.rank() - 1);
    const U64 candidateNumberOfElements = candidateInputRowWidth / config.decimation;
    if (candidateNumberOfElements < 2) {
        JST_ERROR("[MODULE_LINEPLOT] Invalid number of elements ({}), need at least 2.",
                  candidateNumberOfElements);
        return Result::ERROR;
    }

    U64 signalPointScalarCount = 0;
    U64 signalVertexScalarCount = 0;
    U64 signalVertexCount = 0;
    if (!Jetstream::detail::CheckedMultiply(candidateNumberOfElements,
                                            2,
                                            signalPointScalarCount) ||
        !Jetstream::detail::CheckedMultiply(candidateNumberOfElements - 1,
                                            16,
                                            signalVertexScalarCount) ||
        !Jetstream::detail::CheckedMultiply(candidateNumberOfElements - 1,
                                            4,
                                            signalVertexCount) ||
        candidateNumberOfElements > std::numeric_limits<U32>::max() ||
        signalVertexCount > std::numeric_limits<U32>::max() ||
        signalPointScalarCount > maxRenderScalarCount ||
        signalVertexScalarCount > maxRenderScalarCount) {
        JST_ERROR("[MODULE_LINEPLOT] Signal geometry exceeds the supported "
                  "rendering range.");
        return Result::ERROR;
    }

    if (inputTensor.hasAttribute("frequency")) {
        const std::any value = inputTensor.attribute("frequency");
        if (!std::any_cast<F32>(&value)) {
            JST_ERROR("[MODULE_LINEPLOT] Input frequency metadata must have type F32.");
            return Result::ERROR;
        }
    }

    if (inputTensor.hasAttribute("sampleRate")) {
        const std::any value = inputTensor.attribute("sampleRate");
        if (!std::any_cast<F32>(&value)) {
            JST_ERROR("[MODULE_LINEPLOT] Input sample rate metadata must have type F32.");
            return Result::ERROR;
        }
    }

    const U64 candidateNumberOfBatches =
        inputTensor.rank() == 2 ? inputTensor.shape(0) : 1;
    validatedNumberOfElements = candidateNumberOfElements;
    validatedNumberOfBatches = candidateNumberOfBatches;
    validatedInputRowWidth = candidateInputRowWidth;
    validatedNormalizationFactor =
        1.0f / (0.5f * static_cast<F32>(candidateNumberOfBatches));

    return Result::SUCCESS;
}

Result LineplotImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("signal"));

    return Result::SUCCESS;
}

Result LineplotImpl::create() {
    // Get input tensor.

    input = inputs().at("signal").tensor;

    if (input.hasAttribute("frequency")) {
        const std::any value = input.attribute("frequency");
        if (const auto* frequency = std::any_cast<F32>(&value)) {
            JST_DEBUG("[MODULE_LINEPLOT] Input frequency: {:.02f} MHz",
                      *frequency / 1e6f);
        }
    }
    if (input.hasAttribute("sampleRate")) {
        const std::any value = input.attribute("sampleRate");
        if (const auto* sampleRate = std::any_cast<F32>(&value)) {
            JST_DEBUG("[MODULE_LINEPLOT] Input sample rate: {:.02f} MHz",
                      *sampleRate / 1e6f);
        }
    }

    numberOfElements = validatedNumberOfElements;
    numberOfBatches = validatedNumberOfBatches;
    inputRowWidth = validatedInputRowWidth;
    normalizationFactor = validatedNormalizationFactor;

    // Allocate internal buffers.

    // TODO: Restore CUDA/Vulkan zero-copy after adding cross-API synchronization.
    Buffer::Config renderStateConfig{};
    renderStateConfig.hostAccessible = device() == DeviceType::CUDA;

    JST_CHECK(signalPoints.create(device(), DataType::F32, {numberOfElements, 2}, renderStateConfig));
    JST_CHECK(signalVertices.create(device(), DataType::F32, {numberOfElements - 1, 4, 4}, renderStateConfig));

    JST_CHECK(cursorSignalPoint.create(DeviceType::CPU, DataType::F32, {2}));

    return Result::SUCCESS;
}

Result LineplotImpl::destroy() {
    JST_CHECK(destroyPresent());
    return Result::SUCCESS;
}

Result LineplotImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.decimation == decimation &&
        config.numberOfVerticalLines == numberOfVerticalLines &&
        config.numberOfHorizontalLines == numberOfHorizontalLines &&
        config.thickness == thickness) {
        averaging = config.averaging;
        return Result::SUCCESS;
    }

    return Result::RECREATE;
}

Result LineplotImpl::createPresent() {
    auto& window = render();

    if (!window) {
        JST_DEBUG("[MODULE_LINEPLOT] No render window available, skipping present creation.");
        return Result::SUCCESS;
    }

    JST_DEBUG("[MODULE_LINEPLOT] Creating present resources...");

    // Axis component.

    if (!window->hasFont("default_mono")) {
        JST_ERROR("[MODULE_LINEPLOT] Font 'default_mono' not found.");
        return Result::ERROR;
    }

    {
        Render::Components::Axis::Config cfg;
        cfg.numberOfVerticalLines = numberOfVerticalLines;
        cfg.numberOfHorizontalLines = numberOfHorizontalLines;
        cfg.thickness = thickness;
        cfg.font = window->font("default_mono");
        cfg.xTitle = "Frequency (MHz)";
        cfg.yTitle = "Amplitude (dBFS)";
        JST_CHECK(window->build(axis, cfg));
        JST_CHECK(window->bind(axis));
    }

    // Cursor element.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &cursorUniforms;
        cfg.elementByteSize = sizeof(cursorUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(cursorUniformBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenVertices;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(cursorVerticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(U32);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(cursorIndicesBuffer, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {cursorVerticesBuffer, 3},
        };
        cfg.indices = cursorIndicesBuffer;
        JST_CHECK(window->build(cursorVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = cursorVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(drawCursorVertex, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["cursor"];
        cfg.draws = {drawCursorVertex};
        cfg.buffers = {
            {cursorUniformBuffer, Render::Program::Target::VERTEX | Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(cursorProgram, cfg));
    }

    // Signal element.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &signalUniforms;
        cfg.elementByteSize = sizeof(signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(signalUniformBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = signalPoints.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = signalPoints.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(signalPointsBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = signalVertices.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = signalVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX | Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(signalVerticesBuffer, cfg));
    }

    {
        Render::Kernel::Config cfg;
        cfg.gridSize = {numberOfElements - 1, 1, 1};
        cfg.kernels = GlobalKernelsPackage["thicklinestrip"];
        cfg.buffers = {
            {signalUniformBuffer, Render::Kernel::AccessMode::READ},
            {signalPointsBuffer, Render::Kernel::AccessMode::READ},
            {signalVerticesBuffer, Render::Kernel::AccessMode::WRITE},
        };
        JST_CHECK(window->build(signalKernel, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {signalVerticesBuffer, 4},
        };
        JST_CHECK(window->build(signalVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = signalVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLE_STRIP;
        JST_CHECK(window->build(drawSignalVertex, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)TurboLutBytes;
        JST_CHECK(window->build(lutTexture, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draws = {drawSignalVertex};
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {signalUniformBuffer, Render::Program::Target::VERTEX | Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(signalProgram, cfg));
    }

    // Cursor amplitude text label.

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 128;
        cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
        cfg.font = window->font("default_mono");
        cfg.elements = {
            {"amplitude", {.position = {1.0f, 1.0f}}},
        };
        JST_CHECK(window->build(text, cfg));
        JST_CHECK(window->bind(text));
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
        cfg.multisampled = true;
        cfg.clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        JST_CHECK(axis->surfaceUnderlay(cfg));
        cfg.kernels.push_back(signalKernel);
        cfg.programs.push_back(signalProgram);
        cfg.programs.push_back(cursorProgram);
        JST_CHECK(axis->surfaceOverlay(cfg));
        JST_CHECK(text->surface(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    // Initialize variables.

    updateState();

    // Register surface manifest.

    JST_CHECK(surfaceCreateManifest({
        .id = "default",
        .size = interaction.viewSize,
        .surface = framebufferTexture,
    }));

    return Result::SUCCESS;
}

Result LineplotImpl::destroyPresent() {
    auto& window = render();

    if (!window) {
        return Result::SUCCESS;
    }

    JST_CHECK(window->unbind(renderSurface));
    JST_CHECK(window->unbind(text));
    JST_CHECK(window->unbind(axis));
    return Result::SUCCESS;
}

Result LineplotImpl::present() {
    if (!signalPointsBuffer) {
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

    if (interaction.cursorMoved) {
        const auto& ps = axis->paddingScale();
        const F32 cursorNDC_X = interaction.cursorNormalized.x * 2.0f - 1.0f;
        const F32 cursorNDC_Y = interaction.cursorNormalized.y * 2.0f - 1.0f;
        const F32 translation = -2.0f * interaction.offset;
        cursorPos = {
            (cursorNDC_X / interaction.zoom - translation) / ps.x,
            cursorNDC_Y / ps.y
        };
        updateCursorState();
    }

    // Process update flags.

    if (updateSignalPointsFlag) {
        JST_CHECK(signalPointsBuffer->update());
        signalKernel->update();
        updateCursorState();
        updateSignalPointsFlag = false;
    }

    if (updateSignalUniformBufferFlag) {
        JST_CHECK(signalUniformBuffer->update());
        signalKernel->update();
        updateSignalUniformBufferFlag = false;
    }

    if (updateCursorUniformBufferFlag) {
        JST_CHECK(cursorUniformBuffer->update());
        updateCursorUniformBufferFlag = false;
    }

    JST_CHECK(axis->present());
    JST_CHECK(text->present());

    return Result::SUCCESS;
}

void LineplotImpl::updateState() {
    const F32 maxTranslation = std::abs((1.0f / interaction.zoom) - 1.0f);
    const F32 translation = std::clamp(-2.0f * interaction.offset, -maxTranslation, maxTranslation);

    // Update global pixel size.

    pixelSize = {
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y
    };

    // Update axis component (computes paddingScale internally).

    axis->updatePixelSize(pixelSize);
    const auto& paddingScale = axis->paddingScale();

    // Update the signal transform.

    auto signalTransform = glm::mat4(1.0f);

    signalTransform = glm::translate(signalTransform, glm::vec3(translation * interaction.zoom, 0.0f, 0.0f));
    signalTransform = glm::scale(signalTransform, glm::vec3(paddingScale.x, paddingScale.y, 1.0f));

    signalUniforms.transform = signalTransform;
    signalUniforms.thickness[0] = pixelSize.x * thickness * 3.0f;
    signalUniforms.thickness[1] = pixelSize.y * thickness * 3.0f;
    signalUniforms.zoom = interaction.zoom;
    signalUniforms.numberOfPoints = numberOfElements;

    // Clip signal and cursor to the plot area.

    const auto& vs = interaction.viewSize;
    Render::ScissorRect plotRect;
    plotRect.x = static_cast<U32>((1.0f - paddingScale.x) / 2.0f * vs.x);
    plotRect.y = static_cast<U32>((1.0f - paddingScale.y) / 2.0f * vs.y);
    plotRect.width = static_cast<U32>(paddingScale.x * vs.x);
    plotRect.height = static_cast<U32>(paddingScale.y * vs.y);
    signalProgram->scissorRect(plotRect);
    cursorProgram->scissorRect(plotRect);

    // Update the cursor.

    updateCursorState();

    // Schedule the uniform buffers for update.

    updateSignalUniformBufferFlag = true;
}

Result LineplotImpl::readSignalPoint(const U64 index, F32* point) {
    const auto* signalData = signalPoints.data<F32>();
    point[0] = signalData[(index * 2) + 0];
    point[1] = signalData[(index * 2) + 1];
    return Result::SUCCESS;
}

void LineplotImpl::updateCursorState() {
    const auto& paddingScale = axis->paddingScale();

    // Fetch closest cursor plot value.

    const auto stepX = 2.0f / numberOfElements;
    const U64 cursorIndex = std::clamp(static_cast<U64>((cursorPos.x + 1.0f) / stepX), U64{0}, numberOfElements - 1);

    F32* cursorData = static_cast<F32*>(cursorSignalPoint.data());

    if (readSignalPoint(cursorIndex, cursorData) != Result::SUCCESS) {
        return;
    }

    const auto cursorValueX = cursorData[0] * paddingScale.x;
    const auto cursorValueY = cursorData[1] * paddingScale.y;

    F32 centerFrequency = 0.0f;
    F32 inputSampleRate = 0.0f;
    bool hasFrequencyMetadata = false;
    if (input.hasAttribute("frequency") && input.hasAttribute("sampleRate")) {
        const std::any frequency = input.attribute("frequency");
        const std::any sampleRate = input.attribute("sampleRate");
        const auto* typedFrequency = std::any_cast<F32>(&frequency);
        const auto* typedSampleRate = std::any_cast<F32>(&sampleRate);
        if (typedFrequency && typedSampleRate) {
            centerFrequency = *typedFrequency;
            inputSampleRate = *typedSampleRate;
            hasFrequencyMetadata = true;
        }
    }

    const F32 translation = std::clamp(
        -2.0f * interaction.offset,
        -std::abs((1.0f / interaction.zoom) - 1.0f),
        std::abs((1.0f / interaction.zoom) - 1.0f)
    );

    auto transform = glm::mat4(1.0f);

    transform = glm::translate(transform, glm::vec3((cursorValueX + translation) * interaction.zoom, cursorValueY, 0.0f));

    {
        const auto x = pixelSize.x * thickness * 15.0f;
        const auto y = pixelSize.y * thickness * 15.0f;
        transform = glm::scale(transform, glm::vec3(x, y, 1.0f));
    }

    cursorUniforms.transform = transform;

    updateCursorUniformBufferFlag = true;

    // Update tick labels via axis component.

    if (axis) {
        const F32 viewWidthPx = interaction.viewSize.x / interaction.scale;
        const F32 tickSpacingPx = (viewWidthPx * paddingScale.x) / (numberOfVerticalLines - 1);
        const U64 tickStep = std::max(U64{1},
            static_cast<U64>(std::ceil(65.0f / tickSpacingPx)));

        std::vector<std::string> xLabels(numberOfVerticalLines - 2);
        for (U64 i = 1; i < numberOfVerticalLines - 1; i++) {
            if ((i - 1) % tickStep == 0) {
                const F32 tickX = (2.0f * paddingScale.x / (numberOfVerticalLines - 1)) * i - paddingScale.x;
                const F32 normalizedPos = ((tickX / interaction.zoom) - translation) / paddingScale.x;
                if (hasFrequencyMetadata) {
                    const F32 freq =
                        (centerFrequency + normalizedPos * inputSampleRate / 2.0f) /
                        1e6f;
                    xLabels[i - 1] = jst::fmt::format("{:.02f}", freq);
                } else {
                    xLabels[i - 1] = jst::fmt::format("{:.02f}", normalizedPos);
                }
            }
        }

        axis->updateTickLabels(xLabels, {});
    }

    // Update cursor amplitude label.

    if (text) {
        text->updatePixelSize(pixelSize);

        auto element = text->get("amplitude");

        if (hasFrequencyMetadata) {
            const F32 freq =
                (centerFrequency + cursorData[0] * inputSampleRate / 2.0f) /
                1e6f;
            element.fill = jst::fmt::format("{:.03f} MHz, {:.02f} dBFS", freq, cursorData[1]);
        } else {
            element.fill = jst::fmt::format("{:.04f}, {:.04f}", cursorData[0], cursorData[1]);
        }

        element.position = {(cursorValueX + translation) * interaction.zoom + 0.05f, cursorValueY - 0.05f};
        text->update("amplitude", element);
    }
}

}  // namespace Jetstream::Modules
