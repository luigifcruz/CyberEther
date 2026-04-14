#include "module_impl.hh"

#include <algorithm>
#include <any>
#include <cmath>

#include "jetstream/render/utils.hh"
#include "jetstream/constants.hh"
#include "resources/shaders/spectrogram_shaders.hh"

namespace Jetstream::Modules {

Result SpectrogramImpl::validate() {
    const auto& config = *candidate();

    if (config.height == 0 || config.height > 2048) {
        JST_ERROR("[MODULE_SPECTROGRAM] Invalid height value '{}', must be between 1 and 2048.", config.height);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SpectrogramImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("signal"));

    return Result::SUCCESS;
}

Result SpectrogramImpl::create() {
    // Get input tensor.

    input = inputs().at("signal").tensor;

    // Check input rank.

    if (input.rank() > 2) {
        JST_ERROR("[MODULE_SPECTROGRAM] Invalid input rank ({}), expected 1 or 2.", input.rank());
        return Result::ERROR;
    }

    // Calculate parameters.

    const U64 lastAxis = input.rank() - 1;
    numberOfElements = input.shape()[lastAxis];
    numberOfBatches = (input.rank() == 2) ? input.shape()[0] : 1;
    decayFactor = std::pow(kSpectrogramDecayBase, static_cast<F32>(numberOfBatches));

    // Allocate internal buffers.

    JST_CHECK(frequencyBins.create(device(), DataType::F32, {numberOfElements, height}));

    return Result::SUCCESS;
}

Result SpectrogramImpl::destroy() {
    JST_CHECK(destroyPresent());
    return Result::SUCCESS;
}

Result SpectrogramImpl::createPresent() {
    auto& window = render();

    if (!window) {
        JST_DEBUG("[MODULE_SPECTROGRAM] No render window available, skipping present creation.");
        return Result::SUCCESS;
    }

    JST_DEBUG("[MODULE_SPECTROGRAM] Creating present resources...");

    if (!window->hasFont("default_mono")) {
        JST_ERROR("[MODULE_SPECTROGRAM] Font 'default_mono' not found.");
        return Result::ERROR;
    }

    // Fill screen vertices.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenVerticesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenTextureVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 8;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenTextureVerticesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenTextureVerticesBuffer));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(uint32_t);
        cfg.size = 6;
        cfg.target = Render::Buffer::Target::VERTEX_INDICES;
        JST_CHECK(window->build(fillScreenIndicesBuffer, cfg));
        JST_CHECK(window->bind(fillScreenIndicesBuffer));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {fillScreenVerticesBuffer, 3},
            {fillScreenTextureVerticesBuffer, 2},
        };
        cfg.indices = fillScreenIndicesBuffer;
        JST_CHECK(window->build(vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(drawVertex, cfg));
    }

    // Signal buffer.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = frequencyBins.data();
        cfg.size = frequencyBins.size();
        cfg.elementByteSize = sizeof(F32);
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(signalBuffer, cfg));
        JST_CHECK(window->bind(signalBuffer));
    }

    // LUT texture.

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)TurboLutBytes;
        JST_CHECK(window->build(lutTexture, cfg));
        JST_CHECK(window->bind(lutTexture));
    }

    // Uniform buffer.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &signalUniforms;
        cfg.elementByteSize = sizeof(signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(signalUniformBuffer, cfg));
        JST_CHECK(window->bind(signalUniformBuffer));
    }

    // Signal program.

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draws = {drawVertex};
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {signalUniformBuffer, Render::Program::Target::VERTEX |
                                  Render::Program::Target::FRAGMENT},
            {signalBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(signalProgram, cfg));
    }

    // Axis component.

    {
        Render::Components::Axis::Config cfg;
        cfg.font = window->font("default_mono");
        cfg.xTitle = "Frequency (MHz)";
        cfg.yTitle = "Magnitude";
        JST_CHECK(window->build(axis, cfg));
        JST_CHECK(window->bind(axis));
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
        cfg.multisampled = false;
        JST_CHECK(axis->surfaceUnderlay(cfg));
        cfg.programs = {signalProgram};
        JST_CHECK(axis->surfaceOverlay(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    JST_CHECK(updateAxisState());

    // Register surface manifest.

    JST_CHECK(surfaceCreateManifest({
        .id = "default",
        .size = interaction.viewSize,
        .surface = framebufferTexture,
    }));

    return Result::SUCCESS;
}

Result SpectrogramImpl::destroyPresent() {
    auto& window = render();

    if (!window) {
        return Result::SUCCESS;
    }

    JST_CHECK(window->unbind(renderSurface));
    JST_CHECK(window->unbind(axis));
    JST_CHECK(window->unbind(lutTexture));
    JST_CHECK(window->unbind(fillScreenVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenTextureVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenIndicesBuffer));
    JST_CHECK(window->unbind(signalBuffer));
    JST_CHECK(window->unbind(signalUniformBuffer));

    return Result::SUCCESS;
}

Result SpectrogramImpl::present() {
    if (!signalBuffer) {
        return Result::SUCCESS;
    }

    interaction = ProcessSurfaceInteraction(interaction,
                                            surfaceConsumeSurfaceEvents(),
                                            surfaceConsumeMouseEvents());

    if (interaction.viewChanged) {
        renderSurface->size(interaction.viewSize);
        surfaceUpdateManifestSize("default", interaction.viewSize);
        JST_CHECK(updateAxisState());
    }

    signalBuffer->update();

    signalUniforms.width = numberOfElements;
    signalUniforms.height = height;
    signalUniforms.zoom = interaction.zoom;
    signalUniforms.offset = interaction.offset + 0.5f * (1.0f - 1.0f / interaction.zoom);
    signalUniforms.paddingScaleX = axis->paddingScale().x;
    signalUniforms.paddingScaleY = axis->paddingScale().y;

    signalUniformBuffer->update();
    JST_CHECK(axis->present());

    return Result::SUCCESS;
}

Result SpectrogramImpl::updateAxisState() {
    if (!axis) {
        return Result::SUCCESS;
    }

    const Extent2D<F32> pixelSize = {
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y
    };
    JST_CHECK(axis->updatePixelSize(pixelSize));

    const bool hasFreqAttrs = input.hasAttribute("frequency") && input.hasAttribute("sampleRate");
    JST_CHECK(axis->updateTitles(hasFreqAttrs ? "Frequency (MHz)" : "Normalized Frequency",
                                 "Magnitude"));

    const auto& paddingScale = axis->paddingScale();
    const F32 maxTranslation = std::abs((1.0f / interaction.zoom) - 1.0f);
    const F32 translation = std::clamp(-2.0f * interaction.offset, -maxTranslation, maxTranslation);

    const F32 centerFreq = hasFreqAttrs ? std::any_cast<F32>(input.attribute("frequency")) : 0.0f;
    const F32 sampleRate = hasFreqAttrs ? std::any_cast<F32>(input.attribute("sampleRate")) : 0.0f;

    const U64 numVert = axis->getConfig().numberOfVerticalLines;
    std::vector<std::string> xLabels(numVert - 2);
    const F32 viewWidthPx = interaction.viewSize.x / interaction.scale;
    const F32 tickSpacingPx = (viewWidthPx * paddingScale.x) / (numVert - 1);
    const U64 tickStep = std::max(U64{1}, static_cast<U64>(std::ceil(kSpectrogramMinTickSpacingPx /
                                                                      tickSpacingPx)));

    for (U64 i = 1; i < numVert - 1; i++) {
        if ((i - 1) % tickStep == 0) {
            const F32 tickX = (2.0f * paddingScale.x / (numVert - 1)) * i - paddingScale.x;
            const F32 normalizedPos = tickX / (interaction.zoom * paddingScale.x) - translation;
            const F32 labelValue = hasFreqAttrs ?
                (centerFreq + normalizedPos * sampleRate / 2.0f) / 1e6f :
                (normalizedPos + 1.0f) / 2.0f;
            xLabels[i - 1] = jst::fmt::format("{:.02f}", labelValue);
        }
    }

    JST_CHECK(axis->updateTickLabels(xLabels, {}));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
