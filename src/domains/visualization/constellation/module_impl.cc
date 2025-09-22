#include "module_impl.hh"

#include "jetstream/render/utils.hh"

namespace Jetstream::Modules {

namespace {

constexpr F32 ConstellationPointSize = 10.0f;

}  // namespace

Result ConstellationImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("signal"));

    return Result::SUCCESS;
}

Result ConstellationImpl::create() {
    // Get input tensor.

    input = inputs().at("signal").tensor;

    // Check input rank.

    if (input.rank() == 0) {
        JST_ERROR("[MODULE_CONSTELLATION] Input buffer rank is 0.");
        return Result::ERROR;
    }

    if (input.rank() > 2) {
        JST_ERROR("[MODULE_CONSTELLATION] Invalid input rank ({}), expected 1 or 2.",
                  input.rank());
        return Result::ERROR;
    }

    // Calculate number of points.

    numberOfPoints = input.size();

    if (numberOfPoints == 0) {
        JST_ERROR("[MODULE_CONSTELLATION] Input is empty.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ConstellationImpl::destroy() {
    JST_CHECK(destroyPresent());
    return Result::SUCCESS;
}

Result ConstellationImpl::createPresent() {
    auto& window = render();

    if (!window) {
        JST_DEBUG("[MODULE_CONSTELLATION] No render window available, "
                  "skipping present creation.");
        return Result::SUCCESS;
    }

    JST_DEBUG("[MODULE_CONSTELLATION] Creating present resources...");

    // Create shapes component.

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["constellation_points"] = {
            .type = Render::Components::Shapes::Type::CIRCLE,
            .numberOfInstances = numberOfPoints,
            .size = {ConstellationPointSize, ConstellationPointSize},
        };

        JST_CHECK(window->build(shapes, cfg));
        JST_CHECK(window->bind(shapes));
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
        cfg.clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        JST_CHECK(shapes->surface(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    // Register surface manifest.

    JST_CHECK(surfaceCreateManifest({
        .id = "default",
        .size = interaction.viewSize,
        .surface = framebufferTexture,
    }));

    return Result::SUCCESS;
}

Result ConstellationImpl::destroyPresent() {
    auto& window = render();

    if (!window) {
        return Result::SUCCESS;
    }

    JST_CHECK(window->unbind(renderSurface));
    JST_CHECK(window->unbind(shapes));

    return Result::SUCCESS;
}

Result ConstellationImpl::present() {
    if (!shapes) {
        return Result::SUCCESS;
    }

    // Process surface interaction events.

    interaction = ProcessSurfaceInteraction(interaction,
                                            surfaceConsumeSurfaceEvents(),
                                            surfaceConsumeMouseEvents());

    if (interaction.viewChanged) {
        renderSurface->size(interaction.viewSize);
        surfaceUpdateManifestSize("default", interaction.viewSize);

        shapes->updatePixelSize({
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        });
    }

    // Process update flags.

    if (updatePositionsFlag) {
        JST_CHECK(shapes->updatePositions());
        updatePositionsFlag = false;
    }

    JST_CHECK(shapes->present());

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
