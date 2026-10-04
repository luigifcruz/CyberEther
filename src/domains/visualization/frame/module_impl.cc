#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

#include "jetstream/constants.hh"
#include "resources/shaders/frame_shaders.hh"

namespace Jetstream::Modules {

namespace {

bool IsValidFit(const std::string& fit) {
    return fit == "contain" || fit == "cover" || fit == "stretch";
}

bool IsValidInterpolation(const std::string& interpolation) {
    return interpolation == "nearest" ||
           interpolation == "bilinear" ||
           interpolation == "bicubic";
}

const char* ChannelLabel(const U64 channels) {
    switch (channels) {
        case 3: return "RGB";
        case 4: return "RGBA";
        default: return "Scalar";
    }
}

}  // namespace

Result FrameImpl::validate() {
    const auto& config = *candidate();

    if (!IsValidFit(config.fit)) {
        JST_ERROR("[MODULE_FRAME] Invalid fit mode '{}', expected contain, cover, or stretch.",
                  config.fit);
        return Result::ERROR;
    }

    if (!Render::Colormap::Valid(config.colormap)) {
        JST_ERROR("[MODULE_FRAME] Invalid colormap '{}'.", config.colormap);
        return Result::ERROR;
    }

    if (!IsValidInterpolation(config.interpolation)) {
        JST_ERROR("[MODULE_FRAME] Invalid interpolation '{}', expected nearest, bilinear, or bicubic.",
                  config.interpolation);
        return Result::ERROR;
    }

    if (!inputs().contains("frame")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("frame").tensor;
    if (!inputTensor.validShape()) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() < 2 || inputTensor.rank() > 3) {
        JST_ERROR("[MODULE_FRAME] Invalid input rank ({}), expected 2 or 3.",
                  inputTensor.rank());
        return Result::ERROR;
    }

    if (inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const U64 maxElementCount = std::min({
        static_cast<U64>(std::numeric_limits<I32>::max()),
        static_cast<U64>(std::numeric_limits<std::size_t>::max()) / sizeof(F32),
        static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(F32),
    });
    if (inputTensor.size() > maxElementCount) {
        JST_ERROR("[MODULE_FRAME] Frame size exceeds the supported rendering range.");
        return Result::ERROR;
    }

    const U64 channelCount = inputTensor.rank() == 3 ? inputTensor.shape(2) : 1;
    if (channelCount != 1 && channelCount != 3 && channelCount != 4) {
        JST_ERROR("[MODULE_FRAME] Invalid channel count ({}), expected 1, 3, or 4.",
                  channelCount);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FrameImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("frame"));

    return Result::SUCCESS;
}

Result FrameImpl::create() {
    input = inputs().at("frame").tensor;

    height = input.shape()[0];
    width = input.shape()[1];
    channels = (input.rank() == 3) ? input.shape()[2] : 1;

    frameLabel = jst::fmt::format("{} x {}  {}", width, height, ChannelLabel(channels));

    autoRangeMin.store(0.0f);
    autoRangeMax.store(1.0f);

    return Result::SUCCESS;
}

Result FrameImpl::destroy() {
    JST_CHECK(destroyPresent());
    return Result::SUCCESS;
}

Result FrameImpl::reconfigure() {
    const auto& config = *candidate();

    fit = config.fit;
    colormap = config.colormap;
    autoRange = config.autoRange;
    interpolation = config.interpolation;
    xLabel = config.xLabel;
    yLabel = config.yLabel;

    return Result::SUCCESS;
}

Result FrameImpl::createPresent() {
    auto& window = render();

    if (!window) {
        JST_DEBUG("[MODULE_FRAME] No render window available, skipping present creation.");
        return Result::SUCCESS;
    }

    JST_DEBUG("[MODULE_FRAME] Creating present resources...");

    if (!window->hasFont("default_mono")) {
        JST_ERROR("[MODULE_FRAME] Font 'default_mono' not found.");
        return Result::ERROR;
    }

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

    JST_CHECK(lut.create(window, colormap));

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
        cfg.textures = {lut.texture()};
        cfg.buffers = {
            {frameUniformBuffer, Render::Program::Target::VERTEX |
                                 Render::Program::Target::FRAGMENT},
            {frameBuffer, Render::Program::Target::FRAGMENT},
        };
        JST_CHECK(window->build(frameProgram, cfg));
    }

    {
        Render::Components::Axis::Config cfg;
        cfg.showInteriorGrid = false;
        cfg.showFrameTicks = false;
        cfg.yLabelOnRight = true;
        cfg.yLabelOutside = true;
        cfg.font = window->font("default_mono");
        cfg.xTitle = xLabel;
        cfg.yTitle = yLabel;
        JST_CHECK(window->build(axis, cfg));
        JST_CHECK(window->bind(axis));
    }

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
            {"readout",
             {.scale = 0.85f,
              .position = {1.0f, 1.0f},
              .alignment = {2, 0}}},
        };
        JST_CHECK(window->build(text, cfg));
        JST_CHECK(window->bind(text));
    }

    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {
            2.0f / interaction.viewSize.x,
            2.0f / interaction.viewSize.y,
        };
        cfg.elements["selection"] = {
            .type = Render::Components::Shapes::Type::RECT,
            .numberOfInstances = 1,
            .color = {1.0f, 1.0f, 1.0f, 0.15f},
            .position = {-2.0f, -2.0f},
            .size = {0.0f, 0.0f},
            .borderWidth = 1.0f,
            .borderColor = {1.0f, 1.0f, 1.0f, 0.9f},
        };
        JST_CHECK(window->build(selection, cfg));
        JST_CHECK(window->bind(selection));
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
        cfg.programs.push_back(frameProgram);
        JST_CHECK(axis->surfaceUnderlay(cfg));
        JST_CHECK(axis->surfaceOverlay(cfg));
        JST_CHECK(text->surface(cfg));
        JST_CHECK(selection->surface(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    JST_CHECK(updateViewGeometry());
    JST_CHECK(updateAxisState());

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
    JST_CHECK(window->unbind(selection));
    JST_CHECK(window->unbind(text));
    JST_CHECK(window->unbind(axis));
    return Result::SUCCESS;
}

Result FrameImpl::present() {
    auto inputEvents = surfaceConsumeInputEvents();

    interaction = ProcessSurfaceInteraction(interaction,
                                            surfaceConsumeSurfaceEvents(),
                                            {},
                                            {.enableZoom = false,
                                             .enablePan = false,
                                             .enableCursor = false});

    JST_CHECK(updateViewGeometry());
    processInputEvents(inputEvents);

    if (!frameBuffer) {
        return Result::SUCCESS;
    }

    if (interaction.viewChanged) {
        renderSurface->size(interaction.viewSize);
        renderSurface->clearColor(interaction.backgroundColor);
        surfaceUpdateManifestSize("default", interaction.viewSize);
    }

    JST_CHECK(lut.update(colormap));

    JST_CHECK(updateAxisState());
    updateCursorReadout();
    JST_CHECK(updateTextState());
    JST_CHECK(updateSelectionState());

    F32 lower = autoRange ? autoRangeMin.load() : 0.0f;
    F32 upper = autoRange ? autoRangeMax.load() : 1.0f;
    if (!std::isfinite(lower) || !std::isfinite(upper) || upper <= lower) {
        lower = std::isfinite(lower) ? lower : 0.0f;
        upper = lower + 1.0f;
    }

    frameUniforms.width = static_cast<int>(width);
    frameUniforms.height = static_cast<int>(height);
    frameUniforms.channels = static_cast<int>(channels);
    frameUniforms.useLut = (colormap != "grayscale") ? 1 : 0;
    frameUniforms.interpolate = (interpolation == "bilinear") ? 1
                               : (interpolation == "bicubic") ? 2
                                                              : 0;
    frameUniforms.rangeMin = lower;
    frameUniforms.rangeScale = 1.0f / (upper - lower);
    frameUniforms.zoom = view.zoom;
    frameUniforms.centerX = view.center.x;
    frameUniforms.centerY = view.center.y;
    frameUniforms.fitScaleX = view.fitScale.x;
    frameUniforms.fitScaleY = view.fitScale.y;
    frameUniforms.paddingScaleX = axis->paddingScale().x;
    frameUniforms.paddingScaleY = axis->paddingScale().y;

    frameBuffer->update();
    frameUniformBuffer->update();

    JST_CHECK(axis->present());
    JST_CHECK(text->present());
    JST_CHECK(selection->present());

    return Result::SUCCESS;
}

Extent2D<F32> FrameImpl::surfaceToPlot(const Extent2D<F32>& position) const {
    const Extent2D<F32> padding = axis ? axis->paddingScale() : Extent2D<F32>{1.0f, 1.0f};
    return {
        (position.x - 0.5f) / std::max(padding.x, 1e-6f) + 0.5f,
        (position.y - 0.5f) / std::max(padding.y, 1e-6f) + 0.5f,
    };
}

Extent2D<F32> FrameImpl::plotToImage(const Extent2D<F32>& plot) const {
    return {
        ((plot.x - 0.5f) / view.zoom + view.center.x - 0.5f) * view.fitScale.x + 0.5f,
        ((plot.y - 0.5f) / view.zoom + view.center.y - 0.5f) * view.fitScale.y + 0.5f,
    };
}

Extent2D<F32> FrameImpl::clampToFrame(const Extent2D<F32>& plot) const {
    const auto clampAxis = [zoom = view.zoom](const F32 position, const F32 center, const F32 fitScale) {
        // Project the image edges into plot coordinates and intersect them
        // with the viewport so selections exclude letterboxing and cropping.
        const F32 lower = 0.5f + (0.5f - center - 0.5f / fitScale) * zoom;
        const F32 upper = 0.5f + (0.5f - center + 0.5f / fitScale) * zoom;
        return std::clamp(position, std::clamp(lower, 0.0f, 1.0f), std::clamp(upper, 0.0f, 1.0f));
    };
    return {
        clampAxis(plot.x, view.center.x, view.fitScale.x),
        clampAxis(plot.y, view.center.y, view.fitScale.y),
    };
}

void FrameImpl::clampViewCenter() {
    // In view coordinates, the image spans 1 / fitScale and the viewport
    // spans 1 / zoom. Keep smaller images centered and larger ones in bounds.
    const auto limit = [zoom = view.zoom](const F32 fitScale) {
        return std::max(0.0f, 0.5f * ((1.0f / fitScale) - (1.0f / zoom)));
    };
    const F32 limitX = limit(view.fitScale.x);
    const F32 limitY = limit(view.fitScale.y);
    view.center.x = std::clamp(view.center.x, 0.5f - limitX, 0.5f + limitX);
    view.center.y = std::clamp(view.center.y, 0.5f - limitY, 0.5f + limitY);
}

Extent2D<F32> FrameImpl::plotToView(const Extent2D<F32>& plot) const {
    return {
        (plot.x - 0.5f) / view.zoom + view.center.x,
        (plot.y - 0.5f) / view.zoom + view.center.y,
    };
}

F32 FrameImpl::plotDistancePx(const Extent2D<F32>& a, const Extent2D<F32>& b) const {
    const Extent2D<F32> padding = axis ? axis->paddingScale() : Extent2D<F32>{1.0f, 1.0f};
    const F32 dx = (a.x - b.x) * static_cast<F32>(interaction.viewSize.x) * padding.x;
    const F32 dy = (a.y - b.y) * static_cast<F32>(interaction.viewSize.y) * padding.y;
    return std::sqrt(dx * dx + dy * dy);
}

void FrameImpl::zoomToSelection() {
    const auto anchor = clampToFrame(view.selectAnchor);
    const auto current = clampToFrame(view.selectCurrent);

    const F32 padding = axis ? axis->paddingScale().x : 1.0f;
    const F32 widthPx = std::abs(current.x - anchor.x) * static_cast<F32>(interaction.viewSize.x) * padding;
    const F32 heightPx = std::abs(current.y - anchor.y) * static_cast<F32>(interaction.viewSize.y) *
                         (axis ? axis->paddingScale().y : 1.0f);
    if (widthPx < kFrameDragThresholdPx || heightPx < kFrameDragThresholdPx) {
        return;
    }

    const auto a = plotToView(anchor);
    const auto b = plotToView(current);
    const F32 width = std::abs(b.x - a.x);
    const F32 height = std::abs(b.y - a.y);
    if (width < 1e-6f || height < 1e-6f) {
        return;
    }

    view.zoom = std::clamp(std::min(1.0f / width, 1.0f / height), 1.0f, kFrameMaxZoom);
    view.center = {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
    clampViewCenter();
}

Result FrameImpl::updateSelectionState() {
    if (!selection) {
        return Result::SUCCESS;
    }

    JST_CHECK(selection->updatePixelSize({
        2.0f / interaction.viewSize.x,
        2.0f / interaction.viewSize.y,
    }));

    std::span<Extent2D<F32>> positions;
    JST_CHECK(selection->getPositions("selection", positions));
    std::span<Extent2D<F32>> sizes;
    JST_CHECK(selection->getSizes("selection", sizes));

    if (!view.selecting) {
        positions[0] = {-2.0f, -2.0f};
        sizes[0] = {0.0f, 0.0f};
    } else {
        const Extent2D<F32> padding = axis ? axis->paddingScale() : Extent2D<F32>{1.0f, 1.0f};
        const auto anchor = clampToFrame(view.selectAnchor);
        const auto current = clampToFrame(view.selectCurrent);
        const Extent2D<F32> center = {(anchor.x + current.x) * 0.5f, (anchor.y + current.y) * 0.5f};
        positions[0] = {(center.x * 2.0f - 1.0f) * padding.x,
                        (1.0f - center.y * 2.0f) * padding.y};
        sizes[0] = {std::abs(current.x - anchor.x) * static_cast<F32>(interaction.viewSize.x) * padding.x,
                    std::abs(current.y - anchor.y) * static_cast<F32>(interaction.viewSize.y) * padding.y};
    }

    JST_CHECK(selection->updatePositions("selection"));
    JST_CHECK(selection->updateSizes("selection"));
    JST_CHECK(selection->updateProperties("selection", 0.0f, interaction.scale,
                                          {1.0f, 1.0f, 1.0f, 0.9f}));

    return Result::SUCCESS;
}

void FrameImpl::processInputEvents(const std::vector<InputEvent>& events) {
    for (const auto& input : events) {
        const auto mouse = SurfaceMouseEvent(input);
        if (!mouse) continue;
        const auto& event = *mouse;
        switch (event.type) {
            case MouseEventType::Scroll: {
                const F32 oldZoom = view.zoom;
                const F32 newZoom = std::clamp(oldZoom * std::exp(event.scroll.y * kFrameZoomSpeed),
                                               1.0f, kFrameMaxZoom);
                if (!std::isfinite(newZoom) || std::abs(newZoom - oldZoom) < 1e-6f) {
                    break;
                }
                const auto plot = surfaceToPlot(event.position);
                const Extent2D<F32> anchor = {
                    (plot.x - 0.5f) / oldZoom + view.center.x,
                    (plot.y - 0.5f) / oldZoom + view.center.y,
                };
                view.zoom = newZoom;
                view.center = {
                    anchor.x - (plot.x - 0.5f) / newZoom,
                    anchor.y - (plot.y - 0.5f) / newZoom,
                };
                clampViewCenter();
                break;
            }
            case MouseEventType::Click: {
                const auto plot = surfaceToPlot(event.position);
                if (event.button == MouseButton::Left) {
                    view.dragging = true;
                    view.dragAnchor = plot;
                    view.dragCenter = view.center;
                }
                if (event.button == MouseButton::Right) {
                    view.selecting = true;
                    view.selectAnchor = plot;
                    view.selectCurrent = plot;
                }
                break;
            }
            case MouseEventType::Release: {
                if (event.button == MouseButton::Left) {
                    view.dragging = false;
                }
                if (event.button == MouseButton::Right && view.selecting) {
                    view.selectCurrent = surfaceToPlot(event.position);
                    view.selecting = false;
                    if (plotDistancePx(view.selectCurrent, view.selectAnchor) <= kFrameDragThresholdPx) {
                        view.zoom = 1.0f;
                        view.center = {0.5f, 0.5f};
                    } else {
                        zoomToSelection();
                    }
                }
                break;
            }
            case MouseEventType::Move: {
                const auto plot = surfaceToPlot(event.position);
                view.cursor = plot;
                view.hasCursor = true;
                if (view.selecting) {
                    view.selectCurrent = plot;
                }
                if (view.dragging) {
                    view.center = {
                        view.dragCenter.x - (plot.x - view.dragAnchor.x) / view.zoom,
                        view.dragCenter.y - (plot.y - view.dragAnchor.y) / view.zoom,
                    };
                    clampViewCenter();
                }
                break;
            }
            case MouseEventType::Enter: {
                break;
            }
            case MouseEventType::Leave: {
                view.dragging = false;
                view.selecting = false;
                view.hasCursor = false;
                break;
            }
        }
    }
}

void FrameImpl::updateFitScale() {
    const Extent2D<F32> padding = axis ? axis->paddingScale() : Extent2D<F32>{1.0f, 1.0f};
    const F32 plotWidth = std::max(static_cast<F32>(interaction.viewSize.x) * padding.x, 1.0f);
    const F32 plotHeight = std::max(static_cast<F32>(interaction.viewSize.y) * padding.y, 1.0f);
    const F32 viewRatio = plotWidth / plotHeight;
    const F32 imageRatio = static_cast<F32>(std::max<U64>(width, 1)) /
                           static_cast<F32>(std::max<U64>(height, 1));

    if (fit == "stretch") {
        view.fitScale = {1.0f, 1.0f};
        return;
    }

    const bool viewWider = viewRatio > imageRatio;
    if (fit == "cover") {
        view.fitScale = viewWider ? Extent2D<F32>{1.0f, imageRatio / viewRatio}
                                  : Extent2D<F32>{viewRatio / imageRatio, 1.0f};
        return;
    }

    view.fitScale = viewWider ? Extent2D<F32>{viewRatio / imageRatio, 1.0f}
                              : Extent2D<F32>{1.0f, imageRatio / viewRatio};
}

Result FrameImpl::updateViewGeometry() {
    if (axis) {
        const Extent2D<F32> pixelSize = {
            (2.0f * interaction.scale) / interaction.viewSize.x,
            (2.0f * interaction.scale) / interaction.viewSize.y,
        };
        JST_CHECK(axis->updatePixelSize(pixelSize));
    }

    updateFitScale();
    clampViewCenter();

    return Result::SUCCESS;
}

Result FrameImpl::updateAxisState() {
    if (!axis) {
        return Result::SUCCESS;
    }

    JST_CHECK(axis->updateTitles(xLabel, yLabel));

    const auto image = [this](const Extent2D<F32>& plot) {
        return plotToImage(plot);
    };

    auto xFormatter = [image, width = width](const F32 position) {
        const F32 x = image({(position + 1.0f) * 0.5f, 0.5f}).x;
        if (x < 0.0f || x > 1.0f) {
            return std::string(" ");
        }
        return jst::fmt::format("{:.0f}", x * static_cast<F32>(width));
    };

    auto yFormatter = [image, height = height](const F32 position) {
        const F32 y = image({0.5f, (1.0f - position) * 0.5f}).y;
        if (y < 0.0f || y > 1.0f) {
            return std::string(" ");
        }
        return jst::fmt::format("{:.0f}", y * static_cast<F32>(height));
    };

    JST_CHECK(axis->updateTickFormatters(std::move(xFormatter), std::move(yFormatter)));

    return Result::SUCCESS;
}

void FrameImpl::updateCursorReadout() {
    std::string label = "n/a";

    if (view.hasCursor && width > 0 && height > 0) {
        const auto image = plotToImage(view.cursor);
        if (image.x >= 0.0f && image.x < 1.0f && image.y >= 0.0f && image.y < 1.0f) {
            const U64 x = std::min(static_cast<U64>(image.x * static_cast<F32>(width)), width - 1);
            const U64 y = std::min(static_cast<U64>(image.y * static_cast<F32>(height)), height - 1);
            const F32* data = input.data<F32>() + ((y * width) + x) * channels;

            if (channels == 1) {
                label = jst::fmt::format("X {}  Y {}  {:.4g}", x, y, data[0]);
            } else if (channels == 3) {
                label = jst::fmt::format("X {}  Y {}  R {:.3g}  G {:.3g}  B {:.3g}",
                                         x, y, data[0], data[1], data[2]);
            } else {
                label = jst::fmt::format("X {}  Y {}  R {:.3g}  G {:.3g}  B {:.3g}  A {:.3g}",
                                         x, y, data[0], data[1], data[2], data[3]);
            }
        }
    }

    cursorLabel = std::move(label);
}

Result FrameImpl::updateTextState() {
    if (!text || !axis) {
        return Result::SUCCESS;
    }

    const Extent2D<F32> pixelSize = {
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y,
    };
    text->updatePixelSize(pixelSize);

    const auto& padding = axis->paddingScale();
    const F32 tickOffset = axis->getConfig().majorTickLengthPx + 4.0f;
    const bool attached = interaction.placement == SurfacePlacementType::Attached;

    auto header = text->get("header");
    header.position = {-padding.x + pixelSize.x * tickOffset,
                       padding.y - pixelSize.y * tickOffset};
    if (attached) {
        header.fill = " ";
    } else if (std::abs(view.zoom - 1.0f) > 0.01f) {
        header.fill = jst::fmt::format("{}   ZOOM {:.1f}x", frameLabel, view.zoom);
    } else {
        header.fill = frameLabel;
    }
    JST_CHECK(text->update("header", header));

    auto readout = text->get("readout");
    readout.position = {padding.x - pixelSize.x * tickOffset,
                        padding.y - pixelSize.y * tickOffset};
    readout.alignment = {2, 0};
    if (attached || !view.hasCursor) {
        readout.fill = " ";
    } else {
        readout.fill = cursorLabel;
    }
    JST_CHECK(text->update("readout", readout));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
