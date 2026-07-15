#include "module_impl.hh"

#include "jetstream/constants.hh"
#include "resources/shaders/frame_shaders.hh"

namespace Jetstream::Modules {

Result FrameImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("frame"));

    return Result::SUCCESS;
}

Result FrameImpl::create() {
    input = inputs().at("frame").tensor;

    if (input.rank() < 2 || input.rank() > 3) {
        JST_ERROR("[MODULE_FRAME] Invalid input rank ({}), expected 2 or 3.", input.rank());
        return Result::ERROR;
    }

    height = input.shape()[0];
    width = input.shape()[1];
    channels = (input.rank() == 3) ? input.shape()[2] : 1;

    if (width == 0 || height == 0) {
        JST_ERROR("[MODULE_FRAME] Frame dimensions must be non-zero.");
        return Result::ERROR;
    }

    if (channels != 1 && channels != 3 && channels != 4) {
        JST_ERROR("[MODULE_FRAME] Invalid channel count ({}), expected 1, 3, or 4.", channels);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FrameImpl::destroy() {
    JST_CHECK(destroyPresent());
    return Result::SUCCESS;
}

Result FrameImpl::reconfigure() {
    lut = candidate()->lut;
    return Result::SUCCESS;
}

Result FrameImpl::createPresent() {
    auto& window = render();

    if (!window) {
        JST_DEBUG("[MODULE_FRAME] No render window available, skipping present creation.");
        return Result::SUCCESS;
    }

    JST_DEBUG("[MODULE_FRAME] Creating present resources...");

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
        JST_CHECK(window->build(vertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(drawVertex, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = input.data();
        cfg.size = input.size();
        cfg.elementByteSize = sizeof(F32);
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(frameBuffer, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = reinterpret_cast<const uint8_t*>(TurboLutBytes);
        JST_CHECK(window->build(lutTexture, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &frameUniforms;
        cfg.elementByteSize = sizeof(frameUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(frameUniformBuffer, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["signal"];
        cfg.draws = {drawVertex};
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {frameUniformBuffer, Render::Program::Target::VERTEX |
                                 Render::Program::Target::FRAGMENT},
            {frameBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(frameProgram, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = interaction.viewSize;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.multisampled = false;
        cfg.clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        cfg.programs = {frameProgram};
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    JST_CHECK(surfaceCreateManifest({
        .id = "default",
        .size = interaction.viewSize,
        .surface = framebufferTexture,
    }));

    return Result::SUCCESS;
}

Result FrameImpl::destroyPresent() {
    auto& window = render();

    if (!window) {
        return Result::SUCCESS;
    }

    JST_CHECK(window->unbind(renderSurface));
    return Result::SUCCESS;
}

Result FrameImpl::present() {
    if (!frameBuffer) {
        return Result::SUCCESS;
    }

    interaction = ProcessSurfaceInteraction(interaction,
                                            surfaceConsumeSurfaceEvents(),
                                            surfaceConsumeMouseEvents(),
                                            {.enableZoom = false,
                                             .enablePan = false,
                                             .enableCursor = false});

    if (interaction.viewChanged) {
        renderSurface->size(interaction.viewSize);
        renderSurface->clearColor(interaction.backgroundColor);
        surfaceUpdateManifestSize("default", interaction.viewSize);
    }

    frameUniforms.width = static_cast<int>(width);
    frameUniforms.height = static_cast<int>(height);
    frameUniforms.channels = static_cast<int>(channels);
    frameUniforms.useLut = lut ? 1 : 0;

    frameBuffer->update();
    frameUniformBuffer->update();

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
