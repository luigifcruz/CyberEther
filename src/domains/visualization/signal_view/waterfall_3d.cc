#include "waterfall_3d.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

#include "jetstream/constants.hh"
#include "resources/shaders/signal_view_shaders.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kTickCount = 5;
constexpr F32 kLabelMargin = 0.04f;
constexpr F32 kMinLabelOffsetPx = 8.0f;
constexpr F32 kAmplitudeOffsetPx = 12.0f;
constexpr F32 kTitleGapPx = 10.0f;
constexpr F32 kShortEdgePx = 40.0f;
constexpr F32 kTickScale = 0.8f;
constexpr F32 kTitleScale = 0.85f;
constexpr F32 kMinorLineWidthPx = 1.0f;
constexpr F32 kMajorLineWidthPx = 1.6f;
constexpr F32 kMinorLineAlpha = 0.26f;
constexpr F32 kMajorLineAlpha = 0.85f;
constexpr F32 kWallEdgeAlpha = 0.55f;
constexpr F32 kFloorFillAlpha = 0.10f;
constexpr F32 kWallFillAlpha = 0.055f;
constexpr F32 kAmbientLight = 0.4f;
constexpr F32 kAgeFadeStrength = 0.3f;
constexpr F32 kAgeFadeExponent = 1.5f;
constexpr F32 kTraceWidthPx = 2.0f;

std::string TickElementId(const char* prefix, const U64 index) {
    return jst::fmt::format("{}-{}", prefix, index);
}

struct LabelEdge {
    glm::vec3 start;
    glm::vec3 finish;
    glm::vec3 outward;
};

LabelEdge FrequencyEdge(const glm::vec3& eye) {
    const F32 sz = eye.z >= 0.0f ? 1.0f : -1.0f;
    const F32 y = std::abs(eye.z) < 1.0f ? detail::kWaterfall3DHeightScale : 0.0f;
    return {{-1.0f, y, sz}, {1.0f, y, sz}, {0.0f, 0.0f, sz}};
}

LabelEdge TimeEdge(const glm::vec3& eye) {
    const F32 sx = eye.x >= 0.0f ? 1.0f : -1.0f;
    const F32 y = std::abs(eye.x) < 1.0f ? detail::kWaterfall3DHeightScale : 0.0f;
    return {{sx, y, -1.0f}, {sx, y, 1.0f}, {sx, 0.0f, 0.0f}};
}

bool EdgeTouchesColumn(const LabelEdge& edge, const glm::vec2& corner) {
    for (const auto& point : {edge.start, edge.finish}) {
        if (point.x == corner.x && point.z == corner.y) {
            return true;
        }
    }
    return false;
}

}  // namespace

Result SignalViewWaterfall3D::create(const std::shared_ptr<Render::Window>& window,
                                 const U64 binsWidth,
                                 const U64 binsHeight) {
    width = binsWidth;
    height = binsHeight;
    columnCapacity = std::min(width, detail::kWaterfall3DMaxColumns);
    columns = 0;
    heights.assign(columnCapacity * height, 0.0f);
    writeIndex = 0;
    camera = {};
    cameraGoal = {};
    drag = {};
    clockStarted = false;
    sceneDirty = true;
    ticksDirty = true;

    if (!window->hasFont("default_mono")) {
        JST_ERROR("[MODULE_SIGNAL_VIEW] Font 'default_mono' not found.");
        return Result::ERROR;
    }

    // Text labels (tick labels and axis titles).

    {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters = 512;
        cfg.color = {0.86f, 0.88f, 0.92f, 1.0f};
        cfg.font = window->font("default_mono");
        for (const char* prefix : {"freq", "time", "amp"}) {
            for (U64 index = 0; index < kTickCount; ++index) {
                cfg.elements[TickElementId(prefix, index)] = {.scale = kTickScale, .fill = " "};
            }
        }
        cfg.elements["x-title"] = {.scale = kTitleScale, .fill = " "};
        cfg.elements["time-title"] = {.scale = kTitleScale, .fill = " "};
        cfg.elements["amp-title"] = {.scale = kTitleScale, .fill = " "};
        JST_CHECK(window->build(text, cfg));
        JST_CHECK(window->bind(text));
    }

    // Surface mesh.

    meshSlots.resize((std::max(columnCapacity, height) - 1) * 6);
    std::iota(meshSlots.begin(), meshSlots.end(), 0.0f);

    {
        Render::Buffer::Config cfg;
        cfg.buffer = meshSlots.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = meshSlots.size();
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(meshSlotsBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = heights.data();
        cfg.elementByteSize = sizeof(F32);
        cfg.size = heights.size();
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(heightsBuffer, cfg));
    }

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &meshUniforms;
        cfg.elementByteSize = sizeof(meshUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(meshUniformBuffer, cfg));
    }

    {
        Render::Texture::Config cfg;
        cfg.size = {256, 1};
        cfg.buffer = const_cast<U8*>(&TurboLutBytes[0][0]);
        JST_CHECK(window->build(lutTexture, cfg));
    }

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {meshSlotsBuffer, 1},
        };
        JST_CHECK(window->build(meshVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = meshVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        cfg.numberOfInstances = height - 1;
        JST_CHECK(window->build(drawMesh, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["mesh"];
        cfg.draws = {drawMesh};
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {meshUniformBuffer, Render::Program::Target::VERTEX |
                                Render::Program::Target::FRAGMENT},
            {heightsBuffer, Render::Program::Target::VERTEX},
        };
        JST_CHECK(window->build(meshProgram, cfg));
    }

    // Skirts closing the camera-facing sides of the surface.

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {meshSlotsBuffer, 1},
        };
        JST_CHECK(window->build(skirtVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = skirtVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        cfg.numberOfInstances = 2;
        JST_CHECK(window->build(drawSkirt, cfg));
    }

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["skirt"];
        cfg.draws = {drawSkirt};
        cfg.textures = {lutTexture};
        cfg.buffers = {
            {meshUniformBuffer, Render::Program::Target::VERTEX |
                                Render::Program::Target::FRAGMENT},
            {heightsBuffer, Render::Program::Target::VERTEX},
        };
        JST_CHECK(window->build(skirtProgram, cfg));
    }

    // Live trace along the newest row, drawn behind or in front of the mesh.

    for (auto* layer : {&traceBehind, &traceInFront}) {
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {meshSlotsBuffer, 1},
            };
            JST_CHECK(window->build(layer->vertex, cfg));
        }
        {
            Render::Draw::Config cfg;
            cfg.buffer = layer->vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            JST_CHECK(window->build(layer->draw, cfg));
        }
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["trace"];
            cfg.draws = {layer->draw};
            cfg.buffers = {
                {meshUniformBuffer, Render::Program::Target::VERTEX |
                                    Render::Program::Target::FRAGMENT},
                {heightsBuffer, Render::Program::Target::VERTEX},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(layer->program, cfg));
        }
    }

    // Bathtub frame (fills and lines behind and in front of the mesh).

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &frameUniforms;
        cfg.elementByteSize = sizeof(frameUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(frameUniformBuffer, cfg));
    }

    for (auto* layer : {&backdrop, &foreground}) {
        layer->geometry.clear();
        {
            Render::Buffer::Config cfg;
            cfg.buffer = layer->geometry.storage.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = layer->geometry.storage.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(layer->verticesBuffer, cfg));
        }
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {layer->verticesBuffer, detail::kWaterfall3DGeometryStride},
            };
            JST_CHECK(window->build(layer->vertex, cfg));
        }
        {
            Render::Draw::Config cfg;
            cfg.buffer = layer->vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            JST_CHECK(window->build(layer->draw, cfg));
        }
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["frame"];
            cfg.draws = {layer->draw};
            cfg.buffers = {
                {frameUniformBuffer, Render::Program::Target::VERTEX |
                                     Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(layer->program, cfg));
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
        cfg.multisampled = true;
        cfg.clearColor = interaction.backgroundColor;
        cfg.programs = {backdrop.program};
        JST_CHECK(text->surface(cfg));
        cfg.programs.push_back(traceBehind.program);
        cfg.programs.push_back(meshProgram);
        cfg.programs.push_back(skirtProgram);
        cfg.programs.push_back(traceInFront.program);
        cfg.programs.push_back(foreground.program);
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    frameUniforms.color = {0.62f, 0.66f, 0.72f, 1.0f};
    JST_CHECK(frameUniformBuffer->update());

    pixelSize = {
        (2.0f * interaction.scale) / interaction.viewSize.x,
        (2.0f * interaction.scale) / interaction.viewSize.y,
    };
    JST_CHECK(text->updatePixelSize(pixelSize));

    return Result::SUCCESS;
}

Result SignalViewWaterfall3D::destroy(const std::shared_ptr<Render::Window>& window) {
    if (renderSurface) {
        JST_CHECK(window->unbind(renderSurface));
        renderSurface.reset();
    }
    if (text) {
        JST_CHECK(window->unbind(text));
        text.reset();
    }
    return Result::SUCCESS;
}

Result SignalViewWaterfall3D::present(std::vector<SurfaceEvent>&& surfaceEvents,
                                  std::vector<InputEvent>&& inputEvents,
                                  const WaterfallFrame& frame,
                                  const SignalViewWaterfall3DLabels& nextLabels,
                                  bool& viewChanged) {
    viewChanged = false;
    if (!renderSurface) {
        return Result::SUCCESS;
    }

    interaction = ProcessSurfaceInteraction(interaction, std::move(surfaceEvents), {});
    processInputEvents(std::move(inputEvents));

    if (interaction.viewChanged) {
        renderSurface->size(interaction.viewSize);
        renderSurface->clearColor(interaction.backgroundColor);
        pixelSize = {
            (2.0f * interaction.scale) / interaction.viewSize.x,
            (2.0f * interaction.scale) / interaction.viewSize.y,
        };
        JST_CHECK(text->updatePixelSize(pixelSize));
        sceneDirty = true;
        viewChanged = true;
    }

    if (!(nextLabels == labels)) {
        ticksDirty |= nextLabels.hasFrequency != labels.hasFrequency ||
                      nextLabels.centerFrequency != labels.centerFrequency ||
                      nextLabels.sampleRate != labels.sampleRate ||
                      nextLabels.rangeMin != labels.rangeMin ||
                      nextLabels.rangeMax != labels.rangeMax;
        labels = nextLabels;
        sceneDirty = true;
    }

    const bool cameraMoved = advanceCamera();

    bool dataChanged = false;
    const F32 viewWidth = static_cast<F32>(interaction.viewSize.x) /
                          std::max(interaction.scale, 1e-3f);
    const U64 targetColumns = std::min(detail::Waterfall3DMeshColumns(width, viewWidth),
                                       columnCapacity);
    if (targetColumns != columns) {
        JST_CHECK(resizeMesh(frame.bins, targetColumns));
        dataChanged = true;
    } else {
        if (frame.dirty.firstRowCount > 0) {
            JST_CHECK(uploadRows(frame.bins, frame.dirty.startRow, frame.dirty.firstRowCount));
            dataChanged = true;
        }
        if (frame.dirty.secondRowCount > 0) {
            JST_CHECK(uploadRows(frame.bins, 0, frame.dirty.secondRowCount));
            dataChanged = true;
        }
    }
    writeIndex = frame.writeIndex;

    if (ticksDirty) {
        updateTicks();
        ticksDirty = false;
        sceneDirty = true;
    }

    if (sceneDirty || cameraMoved || dataChanged) {
        JST_CHECK(updateScene());
        sceneDirty = false;
    }

    JST_CHECK(text->present());

    return Result::SUCCESS;
}

Result SignalViewWaterfall3D::resizeMesh(const F32* bins, const U64 nextColumns) {
    columns = nextColumns;
    JST_CHECK(uploadRows(bins, 0, height));
    const U64 vertexCount = (columns - 1) * 6;
    JST_CHECK(drawMesh->updateVertexCount(vertexCount));
    JST_CHECK(traceBehind.draw->updateVertexCount(vertexCount));
    JST_CHECK(traceInFront.draw->updateVertexCount(vertexCount));
    JST_CHECK(drawSkirt->updateVertexCount((std::max(columns, height) - 1) * 6));
    sceneDirty = true;
    return Result::SUCCESS;
}

Result SignalViewWaterfall3D::uploadRows(const F32* bins, const U64 startRow, const U64 rowCount) {
    for (U64 row = startRow; row < startRow + rowCount; ++row) {
        detail::Waterfall3DDecimateRow(bins + row * width, width,
                                       heights.data() + row * columns, columns);
    }
    return heightsBuffer->update(startRow * columns, rowCount * columns);
}

void SignalViewWaterfall3D::processInputEvents(std::vector<InputEvent>&& events) {
    const F32 aspect = static_cast<F32>(interaction.viewSize.x) /
                       static_cast<F32>(std::max<U64>(interaction.viewSize.y, 1));

    for (const auto& input : events) {
        const auto mouse = SurfaceMouseEvent(input);
        if (!mouse) continue;
        const auto& event = *mouse;
        const glm::vec2 position(event.position.x, event.position.y);
        switch (event.type) {
            case MouseEventType::Click: {
                if (event.button == MouseButton::Left) {
                    drag.orbiting = true;
                    drag.last = position;
                }
                if (event.button == MouseButton::Right) {
                    drag.panning = true;
                    drag.travel = 0.0f;
                    drag.last = position;
                }
                break;
            }
            case MouseEventType::Release: {
                if (event.button == MouseButton::Left) {
                    drag.orbiting = false;
                }
                if (event.button == MouseButton::Right && drag.panning) {
                    drag.panning = false;
                    if (drag.travel < detail::kWaterfall3DClickTravel) {
                        cameraGoal = {};
                    }
                }
                break;
            }
            case MouseEventType::Move: {
                const glm::vec2 delta = position - drag.last;
                if (drag.orbiting) {
                    cameraGoal.orbit(-delta.x * detail::kWaterfall3DOrbitSpeed,
                                     delta.y * detail::kWaterfall3DTiltSpeed);
                }
                if (drag.panning) {
                    cameraGoal.pan(delta, aspect);
                    drag.travel += glm::length(delta);
                }
                if (drag.orbiting || drag.panning) {
                    drag.last = position;
                }
                break;
            }
            case MouseEventType::Scroll: {
                cameraGoal.dolly(std::exp(-event.scroll.y * detail::kWaterfall3DDollySpeed));
                break;
            }
            case MouseEventType::Enter: {
                break;
            }
            case MouseEventType::Leave: {
                drag.orbiting = false;
                drag.panning = false;
                break;
            }
        }
    }
}

bool SignalViewWaterfall3D::advanceCamera() {
    const auto now = std::chrono::steady_clock::now();
    if (!clockStarted) {
        lastFrameTime = now;
        clockStarted = true;
    }
    const F32 elapsed = std::clamp(
        std::chrono::duration<F32>(now - lastFrameTime).count(), 0.0f, 0.1f);
    lastFrameTime = now;

    const F32 alpha = 1.0f - std::exp(-elapsed / detail::kWaterfall3DSmoothingSeconds);
    const bool moved = camera.approach(cameraGoal, alpha);

    constexpr F32 turn = 2.0f * static_cast<F32>(JST_PI);
    if (std::abs(camera.azimuth) > turn) {
        const F32 shift = std::round(camera.azimuth / turn) * turn;
        camera.azimuth -= shift;
        cameraGoal.azimuth -= shift;
    }

    return moved;
}

void SignalViewWaterfall3D::updateTicks() {
    frequencyTicks = detail::Waterfall3DFrequencyTicks(labels.hasFrequency,
                                                       labels.centerFrequency,
                                                       labels.sampleRate, kTickCount);
    timeTicks = detail::Waterfall3DTimeTicks(height, kTickCount);
    amplitudeTicks = detail::Waterfall3DAmplitudeTicks(labels.rangeMin, labels.rangeMax, kTickCount);
}

Result SignalViewWaterfall3D::updateScene() {
    const F32 aspect = static_cast<F32>(interaction.viewSize.x) /
                       static_cast<F32>(std::max<U64>(interaction.viewSize.y, 1));
    const glm::mat4 viewProjection = camera.projection(aspect) * camera.view();
    const glm::vec3 eye = camera.eye();
    const F32 heightScale = detail::kWaterfall3DHeightScale;
    const F32 sx = eye.x >= 0.0f ? 1.0f : -1.0f;
    const F32 sz = eye.z >= 0.0f ? 1.0f : -1.0f;

    // Mesh uniforms.

    const glm::vec3 light = glm::normalize(camera.up() -
                                           camera.right() * 0.9f +
                                           camera.forward() * 0.35f);
    meshUniforms.viewProjection = viewProjection;
    meshUniforms.cameraCell = glm::vec4(
        (eye.x + 1.0f) * 0.5f * static_cast<F32>(columns - 1),
        (eye.z + 1.0f) * 0.5f * static_cast<F32>(height - 1),
        0.0f, 0.0f);
    meshUniforms.lightDirection = glm::vec4(light, kAmbientLight);
    meshUniforms.fade = glm::vec4(kAgeFadeStrength, kAgeFadeExponent, 0.0f, 0.0f);
    meshUniforms.background = glm::vec4(interaction.backgroundColor.r,
                                        interaction.backgroundColor.g,
                                        interaction.backgroundColor.b,
                                        1.0f);
    meshUniforms.viewport = glm::vec4(static_cast<F32>(interaction.viewSize.x),
                                      static_cast<F32>(interaction.viewSize.y),
                                      kTraceWidthPx * interaction.scale,
                                      0.0f);
    const bool traceNearest = eye.z >= 1.0f;
    traceBehind.program->setEnabled(!traceNearest);
    traceInFront.program->setEnabled(traceNearest);
    meshUniforms.width = static_cast<I32>(columns);
    meshUniforms.height = static_cast<I32>(height);
    meshUniforms.writeIndex = static_cast<I32>(writeIndex);
    meshUniforms.heightScale = heightScale;
    meshUniforms.skirt = glm::vec4(0.0f, 0.0f,
                                   std::abs(eye.x) > 1.0f ? sx : 0.0f,
                                   std::abs(eye.z) > 1.0f ? sz : 0.0f);
    JST_CHECK(meshUniformBuffer->update());

    // Frame geometry.

    const detail::Waterfall3DProjector projector{viewProjection, pixelSize};
    const auto walls = detail::Waterfall3DVisibleWalls(eye);

    backdrop.geometry.clear();
    foreground.geometry.clear();

    const auto addLine = [&](FrameLayer& layer, const glm::vec3& a, const glm::vec3& b,
                             const F32 widthPx, const F32 alpha) {
        glm::vec2 ndcA;
        glm::vec2 ndcB;
        if (projector.projectSegment(a, b, ndcA, ndcB)) {
            layer.geometry.line(ndcA, ndcB, pixelSize, widthPx, alpha);
        }
    };

    const auto addQuad = [&](FrameLayer& layer, const glm::vec3& a, const glm::vec3& b,
                             const glm::vec3& c, const glm::vec3& d, const F32 alpha) {
        const auto ndcA = projector.project(a);
        const auto ndcB = projector.project(b);
        const auto ndcC = projector.project(c);
        const auto ndcD = projector.project(d);
        if (ndcA && ndcB && ndcC && ndcD) {
            layer.geometry.quad(*ndcA, *ndcB, *ndcC, *ndcD, alpha);
        }
    };

    addQuad(backdrop, {-1.0f, 0.0f, -1.0f}, {1.0f, 0.0f, -1.0f},
            {1.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, kFloorFillAlpha);

    const auto wallX = [&](const F32 x, const bool visible) {
        if (!visible) {
            return;
        }
        addQuad(backdrop, {x, 0.0f, -1.0f}, {x, 0.0f, 1.0f},
                {x, heightScale, 1.0f}, {x, heightScale, -1.0f}, kWallFillAlpha);
        for (const auto& tick : timeTicks) {
            addLine(backdrop, {x, 0.0f, tick.position}, {x, heightScale, tick.position},
                    kMinorLineWidthPx, kMinorLineAlpha);
        }
        for (const auto& tick : amplitudeTicks) {
            const F32 y = tick.position * heightScale;
            addLine(backdrop, {x, y, -1.0f}, {x, y, 1.0f},
                    kMinorLineWidthPx, kMinorLineAlpha);
        }
        addLine(backdrop, {x, heightScale, -1.0f}, {x, heightScale, 1.0f},
                kMajorLineWidthPx, kWallEdgeAlpha);
    };

    const auto wallZ = [&](const F32 z, const bool visible) {
        if (!visible) {
            return;
        }
        addQuad(backdrop, {-1.0f, 0.0f, z}, {1.0f, 0.0f, z},
                {1.0f, heightScale, z}, {-1.0f, heightScale, z}, kWallFillAlpha);
        for (const auto& tick : frequencyTicks) {
            addLine(backdrop, {tick.position, 0.0f, z}, {tick.position, heightScale, z},
                    kMinorLineWidthPx, kMinorLineAlpha);
        }
        for (const auto& tick : amplitudeTicks) {
            const F32 y = tick.position * heightScale;
            addLine(backdrop, {-1.0f, y, z}, {1.0f, y, z},
                    kMinorLineWidthPx, kMinorLineAlpha);
        }
        addLine(backdrop, {-1.0f, heightScale, z}, {1.0f, heightScale, z},
                kMajorLineWidthPx, kWallEdgeAlpha);
    };

    wallX(-1.0f, walls.negativeX);
    wallX(1.0f, walls.positiveX);
    wallZ(-1.0f, walls.negativeZ);
    wallZ(1.0f, walls.positiveZ);

    for (const F32 cx : {-1.0f, 1.0f}) {
        for (const F32 cz : {-1.0f, 1.0f}) {
            const bool wallOnX = cx < 0.0f ? walls.negativeX : walls.positiveX;
            const bool wallOnZ = cz < 0.0f ? walls.negativeZ : walls.positiveZ;
            if (wallOnX || wallOnZ) {
                addLine(backdrop, {cx, 0.0f, cz}, {cx, heightScale, cz},
                        kMajorLineWidthPx, kWallEdgeAlpha);
            }
        }
    }

    for (const auto& tick : frequencyTicks) {
        addLine(backdrop, {tick.position, 0.0f, -1.0f}, {tick.position, 0.0f, 1.0f},
                kMinorLineWidthPx, kMinorLineAlpha);
    }
    for (const auto& tick : timeTicks) {
        addLine(backdrop, {-1.0f, 0.0f, tick.position}, {1.0f, 0.0f, tick.position},
                kMinorLineWidthPx, kMinorLineAlpha);
    }
    addLine(backdrop, {-sx, 0.0f, -1.0f}, {-sx, 0.0f, 1.0f},
            kMajorLineWidthPx, kMajorLineAlpha);
    addLine(backdrop, {-1.0f, 0.0f, -sz}, {1.0f, 0.0f, -sz},
            kMajorLineWidthPx, kMajorLineAlpha);

    const bool outsideX = std::abs(eye.x) >= 1.0f;
    const bool outsideZ = std::abs(eye.z) >= 1.0f;
    addLine(outsideX ? foreground : backdrop, {sx, 0.0f, -1.0f}, {sx, 0.0f, 1.0f},
            kMajorLineWidthPx, kMajorLineAlpha);
    addLine(outsideZ ? foreground : backdrop, {-1.0f, 0.0f, sz}, {1.0f, 0.0f, sz},
            kMajorLineWidthPx, kMajorLineAlpha);

    const AmplitudeAxis amplitude = amplitudeAxis(projector, eye);
    {
        const glm::vec2& corner = amplitude.corner;
        const bool wallOnX = corner.x < 0.0f ? walls.negativeX : walls.positiveX;
        const bool wallOnZ = corner.y < 0.0f ? walls.negativeZ : walls.positiveZ;
        if (!wallOnX && !wallOnZ) {
            addLine(outsideX && outsideZ ? foreground : backdrop,
                    {corner.x, 0.0f, corner.y}, {corner.x, heightScale, corner.y},
                    kMajorLineWidthPx, kWallEdgeAlpha);
        }
    }

    JST_CHECK(updateFrameLayer(backdrop));
    JST_CHECK(updateFrameLayer(foreground));

    updateLabels(projector, eye, amplitude);

    return Result::SUCCESS;
}

Result SignalViewWaterfall3D::updateFrameLayer(FrameLayer& layer) {
    if (layer.geometry.used > 0) {
        JST_CHECK(layer.verticesBuffer->update(
            0, layer.geometry.used * detail::kWaterfall3DGeometryStride));
    }
    JST_CHECK(layer.draw->updateVertexCount(layer.geometry.used));
    return Result::SUCCESS;
}

SignalViewWaterfall3D::AmplitudeAxis SignalViewWaterfall3D::amplitudeAxis(
        const detail::Waterfall3DProjector& projector,
        const glm::vec3& eye) const {
    const F32 heightScale = detail::kWaterfall3DHeightScale;
    const LabelEdge edges[] = {FrequencyEdge(eye), TimeEdge(eye)};
    const glm::vec2 scale(pixelSize.x, pixelSize.y);

    struct Candidate {
        glm::vec2 corner;
        F32 extremeX = 0.0f;
        F32 conflict = 0.0f;
        bool valid = false;
    };

    Candidate leftmost;
    Candidate rightmost;
    leftmost.extremeX = std::numeric_limits<F32>::max();
    rightmost.extremeX = std::numeric_limits<F32>::lowest();
    for (const F32 x : {-1.0f, 1.0f}) {
        for (const F32 z : {-1.0f, 1.0f}) {
            const auto bottom = projector.project({x, 0.0f, z});
            const auto top = projector.project({x, heightScale, z});
            if (!bottom || !top) {
                continue;
            }
            const F32 low = std::min(bottom->x, top->x);
            const F32 high = std::max(bottom->x, top->x);
            if (low < leftmost.extremeX) {
                leftmost = {{x, z}, low, 0.0f, true};
            }
            if (high > rightmost.extremeX) {
                rightmost = {{x, z}, high, 0.0f, true};
            }
        }
    }

    const auto measureConflict = [&](Candidate& candidate) {
        if (!candidate.valid) {
            candidate.conflict = 2.0f;
            return;
        }
        for (const auto& edge : edges) {
            if (!EdgeTouchesColumn(edge, candidate.corner)) {
                continue;
            }
            if (edge.start.y > 0.0f) {
                candidate.conflict = std::max(candidate.conflict, 1.0f);
                continue;
            }
            const auto a = projector.project(edge.start);
            const auto b = projector.project(edge.finish);
            if (!a || !b) {
                continue;
            }
            const glm::vec2 direction = (*b - *a) / scale;
            const F32 length = glm::length(direction);
            if (length < kShortEdgePx) {
                candidate.conflict = std::max(candidate.conflict, 1.0f);
                continue;
            }
            candidate.conflict = std::max(candidate.conflict, std::abs(direction.y) / length);
        }
    };
    measureConflict(leftmost);
    measureConflict(rightmost);

    AmplitudeAxis axis;
    if (leftmost.valid && (leftmost.conflict < 0.9f || leftmost.conflict <= rightmost.conflict)) {
        axis.corner = leftmost.corner;
        axis.side = -1.0f;
    } else if (rightmost.valid) {
        axis.corner = rightmost.corner;
        axis.side = 1.0f;
    }
    return axis;
}

void SignalViewWaterfall3D::updateLabels(const detail::Waterfall3DProjector& projector,
                                     const glm::vec3& eye,
                                     const AmplitudeAxis& amplitude) {
    const F32 heightScale = detail::kWaterfall3DHeightScale;
    const glm::vec2 scale(pixelSize.x, pixelSize.y);

    const auto hide = [&](const std::string& id) {
        auto element = text->get(id);
        element.fill = " ";
        text->update(id, element);
    };

    const auto show = [&](const std::string& id, const std::string& fill,
                          const glm::vec2& position, const Extent2D<I32>& alignment) {
        auto element = text->get(id);
        element.fill = fill;
        element.position = {position.x, position.y};
        element.alignment = alignment;
        text->update(id, element);
    };

    const auto alignmentFor = [](const glm::vec2& direction) {
        return Extent2D<I32>{
            direction.x > 0.35f ? 0 : (direction.x < -0.35f ? 2 : 1),
            direction.y > 0.35f ? 2 : (direction.y < -0.35f ? 0 : 1),
        };
    };

    if (interaction.placement == SurfacePlacementType::Attached) {
        for (const char* prefix : {"freq", "time", "amp"}) {
            for (U64 index = 0; index < kTickCount; ++index) {
                hide(TickElementId(prefix, index));
            }
        }
        hide("x-title");
        hide("time-title");
        hide("amp-title");
        return;
    }

    const glm::vec2 center =
        projector.project({0.0f, heightScale * 0.5f, 0.0f}).value_or(glm::vec2(0.0f));
    const F32 tickHeightPx = static_cast<F32>(text->getConfig().font->lineHeight()) * kTickScale;
    const glm::vec3 cornerBase(amplitude.corner.x, 0.0f, amplitude.corner.y);

    const auto labelFloorAxis = [&](const LabelEdge& edge,
                                    const char* prefix,
                                    const std::vector<detail::Waterfall3DTick>& ticks,
                                    const std::string& titleId,
                                    const std::string& title) {
        const auto frame = detail::Waterfall3DAxisFrameFor(projector, edge.start, edge.finish,
                                                           edge.outward, center);
        if (!frame.valid) {
            for (U64 index = 0; index < kTickCount; ++index) {
                hide(TickElementId(prefix, index));
            }
            hide(titleId);
            return;
        }

        F32 widestPx = 0.0f;
        for (const auto& tick : ticks) {
            widestPx = std::max(widestPx, text->advance(tick.label) * kTickScale);
        }
        const U64 stride = detail::Waterfall3DTickStride(frame, pixelSize, kTickCount,
                                                         widestPx, tickHeightPx);

        bool titlePlaced = false;
        for (U64 index = 0; index < kTickCount; ++index) {
            const auto& tick = ticks[index];
            const F32 parameter = (tick.position + 1.0f) * 0.5f;
            const glm::vec3 edgePoint = edge.start + (edge.finish - edge.start) * parameter;
            const auto edgeNdc = projector.project(edgePoint);
            const auto anchorNdc = projector.project(edgePoint + edge.outward * kLabelMargin);
            if (!edgeNdc || !anchorNdc) {
                hide(TickElementId(prefix, index));
                continue;
            }
            const glm::vec2 offset = (*anchorNdc - *edgeNdc) / scale;
            glm::vec2 direction = offset - frame.direction * glm::dot(offset, frame.direction);
            F32 gapPx = glm::length(direction);
            direction = gapPx > 1e-3f ? direction / gapPx : frame.normal;
            gapPx = std::max(gapPx, kMinLabelOffsetPx);
            const glm::vec2 position = *edgeNdc + direction * gapPx * scale;
            auto alignment = alignmentFor(direction);
            if (edgePoint == cornerBase) {
                alignment.y = 0;
            }
            if (index % stride == 0) {
                show(TickElementId(prefix, index), tick.label, position, alignment);
            } else {
                hide(TickElementId(prefix, index));
            }
            if (index == kTickCount / 2) {
                const F32 extentPx = detail::Waterfall3DLabelExtentAlong(direction, alignment,
                                                                         widestPx, tickHeightPx);
                show(titleId, title, position + direction * (extentPx + kTitleGapPx) * scale,
                     alignmentFor(direction));
                titlePlaced = true;
            }
        }
        if (!titlePlaced) {
            hide(titleId);
        }
    };

    const std::string resolvedXLabel =
        !labels.hasFrequency && labels.frequency == "Frequency (MHz)"
            ? "Normalized Frequency"
            : labels.frequency;

    labelFloorAxis(FrequencyEdge(eye), "freq", frequencyTicks, "x-title", resolvedXLabel);
    labelFloorAxis(TimeEdge(eye), "time", timeTicks, "time-title", labels.time);

    const glm::vec2 sideways(amplitude.side, 0.0f);
    const Extent2D<I32> sideAlignment = {amplitude.side < 0.0f ? 2 : 0, 1};
    std::optional<glm::vec2> topPosition;
    for (U64 index = 0; index < kTickCount; ++index) {
        const auto& tick = amplitudeTicks[index];
        const auto anchorNdc = projector.project(
            cornerBase + glm::vec3(0.0f, tick.position * heightScale, 0.0f));
        if (!anchorNdc) {
            hide(TickElementId("amp", index));
            continue;
        }
        const glm::vec2 position = *anchorNdc + sideways * kAmplitudeOffsetPx * scale;
        Extent2D<I32> alignment = sideAlignment;
        if (index == 0) {
            alignment.y = 2;
        }
        show(TickElementId("amp", index), tick.label, position, alignment);
        if (index + 1 == kTickCount) {
            topPosition = position;
        }
    }
    if (topPosition) {
        show("amp-title", labels.amplitude,
             *topPosition + glm::vec2(0.0f, tickHeightPx * 1.4f * pixelSize.y),
             {sideAlignment.x, 2});
    } else {
        hide("amp-title");
    }
}

}  // namespace Jetstream::Modules
