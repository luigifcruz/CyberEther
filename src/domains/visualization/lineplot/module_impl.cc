#include "module_impl.hh"

#include <any>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/render/utils.hh"
#include "jetstream/constants.hh"
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

    if (config.thickness <= 0.0f) {
        JST_ERROR("[MODULE_LINEPLOT] Thickness must be positive.");
        return Result::ERROR;
    }


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
        JST_DEBUG("[MODULE_LINEPLOT] Input frequency: {:.02f} MHz", std::any_cast<F32>(input.attribute("frequency")) / 1e6f);
    }
    if (input.hasAttribute("sampleRate")) {
        JST_DEBUG("[MODULE_LINEPLOT] Input sample rate: {:.02f} MHz", std::any_cast<F32>(input.attribute("sampleRate")) / 1e6f);
    }

    // Check input rank.

    if (input.rank() > 2) {
        JST_ERROR("[MODULE_LINEPLOT] Invalid input rank ({}), expected 1 or 2.", input.rank());
        return Result::ERROR;
    }

    // Calculate parameters.

    const U64 lastAxis = input.rank() - 1;
    numberOfElements = input.shape()[lastAxis] / decimation;
    numberOfBatches = (input.rank() == 2) ? input.shape()[0] : 1;
    normalizationFactor = 1.0f / (0.5f * numberOfBatches);

    // Check shape.

    if (numberOfElements < 2) {
        JST_ERROR("[MODULE_LINEPLOT] Invalid number of elements ({}), need at least 2.", numberOfElements);
        return Result::ERROR;
    }

    // Allocate internal buffers.

    JST_CHECK(signalPoints.create(device(), DataType::F32, {numberOfElements, 2}));
    JST_CHECK(signalVertices.create(device(), DataType::F32, {numberOfElements - 1, 4, 4}));

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
        JST_CHECK(window->bind(cursorUniformBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenVertices;
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(cursorVerticesBuffer, cfg));
        JST_CHECK(window->bind(cursorVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(U32);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(cursorIndicesBuffer, cfg));
        JST_CHECK(window->bind(cursorIndicesBuffer));
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
        JST_CHECK(window->bind(signalUniformBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = signalPoints.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = signalPoints.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(signalPointsBuffer, cfg));
        JST_CHECK(window->bind(signalPointsBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = signalVertices.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = signalVertices.size();
        cfg.target = Render::Buffer::Target::VERTEX | Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(signalVerticesBuffer, cfg));
        JST_CHECK(window->bind(signalVerticesBuffer));
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
        JST_CHECK(window->bind(lutTexture));
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
            {"amplitude", {1.0f, {1.0f, 1.0f}, {0, 0}, 0.0f, ""}},
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
    JST_CHECK(window->unbind(lutTexture));
    JST_CHECK(window->unbind(signalPointsBuffer));
    JST_CHECK(window->unbind(signalVerticesBuffer));
    JST_CHECK(window->unbind(signalUniformBuffer));
    JST_CHECK(window->unbind(cursorUniformBuffer));
    JST_CHECK(window->unbind(cursorVerticesBuffer));
    JST_CHECK(window->unbind(cursorIndicesBuffer));

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
        signalPointsBuffer->update();
        signalKernel->update();
        updateCursorState();
        updateSignalPointsFlag = false;
    }

    if (updateSignalUniformBufferFlag) {
        signalUniformBuffer->update();
        signalKernel->update();
        updateSignalUniformBufferFlag = false;
    }

    if (updateCursorUniformBufferFlag) {
        cursorUniformBuffer->update();
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

void LineplotImpl::updateCursorState() {
    const auto& paddingScale = axis->paddingScale();

    // Fetch closest cursor plot value.

    const auto stepX = 2.0f / numberOfElements;
    const U64 cursorIndex = std::clamp(static_cast<U64>((cursorPos.x + 1.0f) / stepX), U64{0}, numberOfElements - 1);

    F32* signalData = static_cast<F32*>(signalPoints.data());
    F32* cursorData = static_cast<F32*>(cursorSignalPoint.data());

    cursorData[0] = signalData[(cursorIndex * 2) + 0];
    cursorData[1] = signalData[(cursorIndex * 2) + 1];

    const auto cursorValueX = cursorData[0] * paddingScale.x;
    const auto cursorValueY = cursorData[1] * paddingScale.y;

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

        const bool hasFreqAttrs = input.hasAttribute("frequency") && input.hasAttribute("sampleRate");
        const F32 centerFreq = hasFreqAttrs ? std::any_cast<F32>(input.attribute("frequency")) : 0.0f;
        const F32 sampleRate = hasFreqAttrs ? std::any_cast<F32>(input.attribute("sampleRate")) : 0.0f;

        std::vector<std::string> xLabels(numberOfVerticalLines - 2);
        for (U64 i = 1; i < numberOfVerticalLines - 1; i++) {
            if ((i - 1) % tickStep == 0) {
                const F32 tickX = (2.0f * paddingScale.x / (numberOfVerticalLines - 1)) * i - paddingScale.x;
                const F32 normalizedPos = ((tickX / interaction.zoom) - translation) / paddingScale.x;
                if (hasFreqAttrs) {
                    const F32 freq = (centerFreq + normalizedPos * sampleRate / 2.0f) / 1e6f;
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

        const bool hasFreqAttrs = input.hasAttribute("frequency") && input.hasAttribute("sampleRate");
        if (hasFreqAttrs) {
            const F32 centerFreq = std::any_cast<F32>(input.attribute("frequency"));
            const F32 sampleRate = std::any_cast<F32>(input.attribute("sampleRate"));
            const F32 freq = (centerFreq + cursorData[0] * sampleRate / 2.0f) / 1e6f;
            element.fill = jst::fmt::format("{:.03f} MHz, {:.02f} dBFS", freq, cursorData[1]);
        } else {
            element.fill = jst::fmt::format("{:.04f}, {:.04f}", cursorData[0], cursorData[1]);
        }

        element.position = {(cursorValueX + translation) * interaction.zoom + 0.05f, cursorValueY - 0.05f};
        text->update("amplitude", element);
    }
}

}  // namespace Jetstream::Modules
