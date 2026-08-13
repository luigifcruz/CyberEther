#include "module_impl.hh"

#include <algorithm>
#include <any>
#include <cmath>
#include <limits>

#include "jetstream/render/utils.hh"
#include "jetstream/constants.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/tools/numeric.hh"
#include "resources/shaders/spectrogram_shaders.hh"

namespace Jetstream::Modules {

Result SpectrogramImpl::validate() {
    validatedNumberOfElements = 0;
    validatedNumberOfBatches = 0;
    validatedInputSampleStride = 0;
    validatedInputBatchStride = 0;

    const auto& config = *candidate();

    if (config.height == 0 || config.height > 2048) {
        JST_ERROR("[MODULE_SPECTROGRAM] Invalid height value '{}', must be between 1 and 2048.", config.height);
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
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_SPECTROGRAM] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    if (axes.channel) {
        JST_ERROR("[MODULE_SPECTROGRAM] channelAxis is not supported.");
        return Result::ERROR;
    }

    for (Index axis = 0; axis < inputTensor.rank(); ++axis) {
        if (axis != *axes.sample && (!axes.batch || axis != *axes.batch)) {
            JST_ERROR("[MODULE_SPECTROGRAM] Unsupported auxiliary input axis {}. "
                      "Every dimension must be sampleAxis or batchAxis.", axis);
            return Result::ERROR;
        }
    }

    const U64 width = inputTensor.shape(*axes.sample);
    U64 renderBinCount = 0;
    const U64 maxRenderBinCount = std::min({
        static_cast<U64>(std::numeric_limits<U32>::max()),
        static_cast<U64>(std::numeric_limits<std::size_t>::max()) / sizeof(F32),
        static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(F32),
    });
    if (!detail::CheckedMultiply(width, config.height, renderBinCount) ||
        renderBinCount > maxRenderBinCount) {
        JST_ERROR("[MODULE_SPECTROGRAM] Render bin count exceeds the supported range.");
        return Result::ERROR;
    }

    validatedNumberOfElements = width;
    validatedNumberOfBatches = axes.batch ? inputTensor.shape(*axes.batch) : 1;
    validatedInputSampleStride = inputTensor.stride(*axes.sample);
    validatedInputBatchStride = axes.batch ? inputTensor.stride(*axes.batch) : 0;

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

    // Calculate parameters.

    numberOfElements = validatedNumberOfElements;
    numberOfBatches = validatedNumberOfBatches;
    inputSampleStride = validatedInputSampleStride;
    inputBatchStride = validatedInputBatchStride;
    decayFactor = std::pow(kSpectrogramDecayBase, static_cast<F32>(numberOfBatches));

    // Allocate internal buffers.

    JST_CHECK(frequencyBins.create(device(), DataType::F32, {numberOfElements, height}));

    signalUniforms.width = static_cast<U32>(numberOfElements);
    signalUniforms.height = static_cast<U32>(height);

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
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenTextureVertices;
        cfg.elementByteSize = sizeof(float);
        cfg.size = 8;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(fillScreenTextureVerticesBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &FillScreenIndices;
        cfg.elementByteSize = sizeof(uint32_t);
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
    }

    // LUT texture.

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = (uint8_t*)TurboLutBytes;
        JST_CHECK(window->build(lutTexture, cfg));
    }

    // Uniform buffer.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &signalUniforms;
        cfg.elementByteSize = sizeof(signalUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(signalUniformBuffer, cfg));
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
        cfg.showInteriorGrid = false;
        cfg.showFrameTicks = true;
        cfg.font = window->font("default_mono");
        cfg.xTitle = xLabel;
        cfg.yTitle = yLabel;
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
        cfg.programs.push_back(signalProgram);
        JST_CHECK(axis->surfaceUnderlay(cfg));
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
    const std::string resolvedXLabel = !hasFreqAttrs && xLabel == "Frequency (MHz)"
        ? "Normalized Frequency"
        : xLabel;
    JST_CHECK(axis->updateTitles(resolvedXLabel, yLabel));

    const F32 maxTranslation = std::abs((1.0f / interaction.zoom) - 1.0f);
    const F32 translation = std::clamp(-2.0f * interaction.offset, -maxTranslation, maxTranslation);

    const F32 centerFreq = hasFreqAttrs ? std::any_cast<F32>(input.attribute("frequency")) : 0.0f;
    const F32 sampleRate = hasFreqAttrs ? std::any_cast<F32>(input.attribute("sampleRate")) : 0.0f;

    auto xFormatter = [hasFreqAttrs, centerFreq, sampleRate,
                       zoom = interaction.zoom, translation](const F32 position) {
        const F32 normalizedPos = position / zoom - translation;
        const F32 labelValue = hasFreqAttrs ?
            (centerFreq + normalizedPos * sampleRate / 2.0f) / 1e6f :
            (normalizedPos + 1.0f) / 2.0f;
        return jst::fmt::format("{:.02f}", labelValue);
    };

    axis->setShowFrameTicks(interaction.placement != SurfacePlacementType::Attached);

    JST_CHECK(axis->updateTickFormatters(std::move(xFormatter)));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
