#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include "jetstream/render/utils.hh"

namespace Jetstream::Modules {

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

    if (!window->hasFont("default_mono")) {
        JST_ERROR("[MODULE_CONSTELLATION] Font 'default_mono' not found.");
        return Result::ERROR;
    }

    // Create axis component.

    {
        Render::Components::Axis::Config cfg;
        cfg.numberOfVerticalLines = 5;
        cfg.numberOfHorizontalLines = 5;
        cfg.font = window->font("default_mono");
        cfg.xTitle = "In-Phase";
        cfg.yTitle = "Quadrature";
        JST_CHECK(window->build(axis, cfg));
        JST_CHECK(window->bind(axis));
    }

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
            .size = {kConstellationPointSize, kConstellationPointSize},
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
        JST_CHECK(axis->surfaceUnderlay(cfg));
        JST_CHECK(shapes->surface(cfg));
        JST_CHECK(axis->surfaceOverlay(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    JST_CHECK(updateAxisState());
    JST_CHECK(updatePointPositions());

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
    JST_CHECK(window->unbind(axis));
    JST_CHECK(window->unbind(shapes));

    return Result::SUCCESS;
}

Result ConstellationImpl::present() {
    if (!shapes) {
        return Result::SUCCESS;
    }

    // Process surface interaction events.

    auto surfaceEvents = surfaceConsumeSurfaceEvents();
    auto mouseEvents = surfaceConsumeMouseEvents();

    interaction = ProcessSurfaceInteraction(interaction,
                                            std::move(surfaceEvents),
                                            {},
                                            {.enableZoom = false,
                                             .enablePan = false,
                                             .enableCursor = false});
    interaction.offset = 0.0f;

    for (const auto& event : mouseEvents) {
        if (event.type == MouseEventType::Scroll) {
            const F32 newZoom = interaction.zoom * std::exp(event.scroll.y * kConstellationZoomSpeed);

            if (std::isfinite(newZoom) && newZoom > std::numeric_limits<F32>::min()) {
                interaction.viewChanged = interaction.viewChanged ||
                                          std::abs(newZoom - interaction.zoom) > 1e-6f;
                interaction.zoom = newZoom;
            }
        }

        if (event.type == MouseEventType::Click && event.button == MouseButton::Right) {
            interaction.viewChanged = interaction.viewChanged || std::abs(interaction.zoom - 1.0f) > 1e-6f;
            interaction.zoom = 1.0f;
        }
    }

    if (interaction.viewChanged) {
        renderSurface->size(interaction.viewSize);
        surfaceUpdateManifestSize("default", interaction.viewSize);
        JST_CHECK(updateAxisState());
        JST_CHECK(updatePointPositions());
    }

    // Process update flags.

    if (updatePositionsFlag) {
        JST_CHECK(shapes->updatePositions());
        updatePositionsFlag = false;
    }

    JST_CHECK(shapes->present());
    JST_CHECK(axis->present());

    return Result::SUCCESS;
}

Result ConstellationImpl::updateAxisState() {
    if (!axis || !shapes) {
        return Result::SUCCESS;
    }

    const Extent2D<F32> pixelSize = {
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y,
    };
    JST_CHECK(axis->updatePixelSize(pixelSize));
    JST_CHECK(shapes->updatePixelSize(pixelSize));

    const U64 numVert = axis->getConfig().numberOfVerticalLines;
    std::vector<std::string> xLabels(numVert - 2);

    for (U64 i = 1; i < numVert - 1; i++) {
        const F32 value = ((2.0f * static_cast<F32>(i) / static_cast<F32>(numVert - 1)) - 1.0f) /
                          interaction.zoom;
        xLabels[i - 1] = jst::fmt::format("{:.2f}", value);
    }

    JST_CHECK(axis->updateTickLabels(xLabels, {}));

    const auto& paddingScale = axis->paddingScale();
    const auto& vs = interaction.viewSize;
    Render::ScissorRect plotRect;
    plotRect.x = static_cast<U32>((1.0f - paddingScale.x) / 2.0f * vs.x);
    plotRect.y = static_cast<U32>((1.0f - paddingScale.y) / 2.0f * vs.y);
    plotRect.width = static_cast<U32>(paddingScale.x * vs.x);
    plotRect.height = static_cast<U32>(paddingScale.y * vs.y);
    JST_CHECK(shapes->updateScissorRect(plotRect));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
