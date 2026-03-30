#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include "jetstream/render/utils.hh"
#include "jetstream/constants.hh"
#include "resources/shaders/map_shaders.hh"

namespace Jetstream::Modules {

Result AdsbImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));

    JST_CHECK(defineInterfaceInput("signal"));
    return Result::SUCCESS;
}

Result AdsbImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    if (input.size() == 0) {
        JST_ERROR("[MODULE_ADSB] Input tensor cannot be empty.");
        return Result::ERROR;
    }

    if (input.hasAttribute("frequency")) {
        JST_INFO("[MODULE_ADSB] Input frequency: {:.2f} MHz",
                 std::any_cast<F32>(input.attribute("frequency")) / 1e6f);
    }

    if (input.hasAttribute("sampleRate")) {
        JST_INFO("[MODULE_ADSB] Input sample rate: {:.2f} MHz",
                 std::any_cast<F32>(input.attribute("sampleRate")) / 1e6f);
    }

    // Allocate output tensors.

    JST_CHECK(aircraft.create(device(), DataType::F32, {maxAircraft, 4}));
    JST_CHECK(aircraftCount.create(device(), DataType::U64, {1}));

    {
        std::lock_guard<std::mutex> lock(aircraftMutex);
        aircraftMap.clear();
        hoveredFlightId.clear();
        hoveredFlightNdc = {0.0f, 0.0f};
        hasLastCursorSample = false;
        lastCursorSample = {0.0f, 0.0f};
    }

    surfaceInteraction.viewSize = mapInteraction.viewSize;
    surfaceInteraction.scale = mapInteraction.scale;

    updateOutputTensors();

    return Result::SUCCESS;
}

std::string AdsbImpl::getAircraftTable() {
    std::lock_guard<std::mutex> lock(aircraftMutex);

    if (aircraftMap.empty()) {
        return "No aircraft detected.";
    }

    std::string table;
    table += "ICAO\tCallsign\tAlt (ft)\tSpeed (kt)\tHdg\tLat\tLon\n";

    for (const auto& [icao, ac] : aircraftMap) {
        table += jst::fmt::format("{:06X}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            icao,
            ac.hasCallsign ? ac.callsign : "-",
            ac.hasAltitude ? jst::fmt::format("{}", ac.altitude) : "-",
            ac.hasVelocity ? jst::fmt::format("{:.0f}", ac.speed) : "-",
            ac.hasVelocity ? jst::fmt::format("{:.0f}", ac.heading) : "-",
            ac.hasPosition ? jst::fmt::format("{:.4f}", ac.latitude) : "-",
            ac.hasPosition ? jst::fmt::format("{:.4f}", ac.longitude) : "-");
    }

    return table;
}

void AdsbImpl::updateOutputTensors() {
    std::lock_guard<std::mutex> lock(aircraftMutex);

    F32* data = static_cast<F32*>(aircraft.data());
    U64* count = static_cast<U64*>(aircraftCount.data());

    U64 idx = 0;
    for (const auto& [icao, ac] : aircraftMap) {
        if (!ac.hasPosition || idx >= maxAircraft) {
            continue;
        }
        data[idx * 4 + 0] = static_cast<F32>(ac.latitude);
        data[idx * 4 + 1] = static_cast<F32>(ac.longitude);
        data[idx * 4 + 2] = ac.heading;
        data[idx * 4 + 3] = static_cast<F32>(ac.altitude);
        idx++;
    }

    // Zero remaining entries.
    for (U64 i = idx; i < maxAircraft; ++i) {
        data[i * 4 + 0] = 0.0f;
        data[i * 4 + 1] = 0.0f;
        data[i * 4 + 2] = 0.0f;
        data[i * 4 + 3] = 0.0f;
    }

    count[0] = idx;
}

// Rendering lifecycle.

Result AdsbImpl::destroy() {
    JST_CHECK(destroyPresent());
    return Result::SUCCESS;
}

Result AdsbImpl::createPresent() {
    auto& window = render();

    if (!window) {
        JST_DEBUG("[MODULE_ADSB] No render window available, skipping "
                  "present creation.");
        return Result::SUCCESS;
    }

    JST_DEBUG("[MODULE_ADSB] Creating present resources...");

    // Create GeoMap render component.

    {
        Render::Components::GeoMap::Config cfg;
        geoMapComponent = std::make_unique<Render::Components::GeoMap>(cfg);
        JST_CHECK(geoMapComponent->create(window.get()));
    }

    // Fill screen geometry (for aircraft pass).

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

    // Aircraft storage buffer.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = aircraft.data();
        cfg.size = aircraft.size();
        cfg.elementByteSize = sizeof(F32);
        cfg.target = Render::Buffer::Target::STORAGE;
        cfg.enableZeroCopy = false;
        JST_CHECK(window->build(aircraftBuffer, cfg));
        JST_CHECK(window->bind(aircraftBuffer));
    }

    // Aircraft vertex + draw.

    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {fillScreenVerticesBuffer, 3},
            {fillScreenTextureVerticesBuffer, 2},
        };
        cfg.indices = fillScreenIndicesBuffer;
        JST_CHECK(window->build(aircraftVertex, cfg));
    }

    {
        Render::Draw::Config cfg;
        cfg.buffer = aircraftVertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        JST_CHECK(window->build(drawAircraftDots, cfg));
    }

    // Aircraft uniform buffer.

    {
        Render::Buffer::Config cfg;
        cfg.buffer = &aircraftUniforms;
        cfg.elementByteSize = sizeof(aircraftUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(aircraftUniformBuffer, cfg));
        JST_CHECK(window->bind(aircraftUniformBuffer));
    }

    // Aircraft dots program.

    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["aircraft"];
        cfg.draws = {drawAircraftDots};
        cfg.buffers = {
            {aircraftUniformBuffer, Render::Program::Target::VERTEX |
                                    Render::Program::Target::FRAGMENT},
            {aircraftBuffer, Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(aircraftProgram, cfg));
    }

    // Track rendering resources.

    {
        // Static quad vertices.
        static F32 trackQuadVertices[] = {
            0.0f, -1.0f,
            1.0f, -1.0f,
            0.0f,  1.0f,
            0.0f,  1.0f,
            1.0f, -1.0f,
            1.0f,  1.0f,
        };

        // Quad vertex buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = trackQuadVertices;
            cfg.elementByteSize = sizeof(F32);
            cfg.size = 12;  // 6 vertices * 2 components
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(trackQuadBuffer, cfg));
            JST_CHECK(window->bind(trackQuadBuffer));
        }

        // Instance buffer (dynamic, max capacity).
        const U64 maxSegments = maxAircraft * maxTrackPoints;
        trackSegments.resize(maxSegments * 4, 0.0f);

        {
            Render::Buffer::Config cfg;
            cfg.buffer = trackSegments.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = trackSegments.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            cfg.enableZeroCopy = false;
            JST_CHECK(window->build(trackInstanceBuffer, cfg));
            JST_CHECK(window->bind(trackInstanceBuffer));
        }

        // Uniform buffer.
        trackGpuUniforms.centerLon = mapInteraction.centerLon;
        trackGpuUniforms.centerLat = mapInteraction.centerLat;
        trackGpuUniforms.zoom = mapInteraction.zoom;
        trackGpuUniforms.aspectRatio = 1.0f;
        trackGpuUniforms.surfaceScale = mapInteraction.scale;
        trackGpuUniforms.lineWidth = 5.0f;
        trackGpuUniforms.colorR = 0.8f;
        trackGpuUniforms.colorG = 0.4f;
        trackGpuUniforms.colorB = 0.2f;
        trackGpuUniforms.viewportWidth = static_cast<F32>(mapInteraction.viewSize.x);
        trackGpuUniforms.viewportHeight = static_cast<F32>(mapInteraction.viewSize.y);

        {
            Render::Buffer::Config cfg;
            cfg.buffer = &trackGpuUniforms;
            cfg.elementByteSize = sizeof(trackGpuUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(trackUniformBuffer, cfg));
            JST_CHECK(window->bind(trackUniformBuffer));
        }

        // Vertex config: quad vertices + instance data.
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {trackQuadBuffer, 2},
            };
            cfg.instances = {
                {trackInstanceBuffer, 4},
            };
            JST_CHECK(window->build(trackVertex, cfg));
        }

        // Draw config: triangles with instancing.
        {
            Render::Draw::Config cfg;
            cfg.buffer = trackVertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            cfg.numberOfInstances = maxSegments;
            JST_CHECK(window->build(trackDraw, cfg));
        }

        // Program using map shaders.
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["map"];
            cfg.draws = {trackDraw};
            cfg.buffers = {
                {trackUniformBuffer, Render::Program::Target::VERTEX |
                                     Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(trackProgram, cfg));
        }
    }

    // Hover label text.

    {
        if (window->hasFont("default_mono")) {
            Render::Components::Text::Config cfg;
            cfg.maxCharacters = 64;
            cfg.color = {1.0f, 1.0f, 1.0f, 1.0f};
            cfg.font = window->font("default_mono");
            cfg.elements = {
                {"flight", {0.9f, {0.0f, 0.0f}, {0, 2}, 0.0f, ""}},
            };
            JST_CHECK(window->build(hoverText, cfg));
            JST_CHECK(window->bind(hoverText));
        } else {
            JST_WARN("[MODULE_ADSB] default_mono font not available, hover label disabled.");
        }
    }

    // Framebuffer texture.

    {
        Render::Texture::Config cfg;
        cfg.size = mapInteraction.viewSize;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }

    // Surface (GeoMap first, then tracks, then aircraft on top).

    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.multisampled = false;

        JST_CHECK(geoMapComponent->surface(cfg));
        cfg.programs.push_back(trackProgram);
        cfg.programs.push_back(aircraftProgram);
        if (hoverText) {
            JST_CHECK(hoverText->surface(cfg));
        }

        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }

    // Register surface manifest.

    JST_CHECK(surfaceCreateManifest({
        .id = "default",
        .size = mapInteraction.viewSize,
        .surface = framebufferTexture,
        .forwardMouseEvents = true,
    }));

    return Result::SUCCESS;
}

Result AdsbImpl::destroyPresent() {
    auto& window = render();

    if (!window) {
        return Result::SUCCESS;
    }

    JST_CHECK(window->unbind(renderSurface));
    JST_CHECK(window->unbind(fillScreenVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenTextureVerticesBuffer));
    JST_CHECK(window->unbind(fillScreenIndicesBuffer));
    JST_CHECK(window->unbind(aircraftBuffer));
    JST_CHECK(window->unbind(aircraftUniformBuffer));
    JST_CHECK(window->unbind(trackQuadBuffer));
    JST_CHECK(window->unbind(trackInstanceBuffer));
    JST_CHECK(window->unbind(trackUniformBuffer));
    if (hoverText) {
        JST_CHECK(window->unbind(hoverText));
    }

    if (geoMapComponent) {
        JST_CHECK(geoMapComponent->destroy(window.get()));
    }

    return Result::SUCCESS;
}

Result AdsbImpl::present() {
    if (!aircraftBuffer) {
        return Result::SUCCESS;
    }

    processMapInteraction(surfaceConsumeSurfaceEvents(),
                          surfaceConsumeMouseEvents());

    if (mapInteraction.viewChanged) {
        renderSurface->size(mapInteraction.viewSize);
        surfaceUpdateManifestSize("default", mapInteraction.viewSize);
        mapInteraction.viewChanged = false;
    }

    // Update GeoMap component uniforms.
    const F32 w = static_cast<F32>(mapInteraction.viewSize.x);
    const F32 h = static_cast<F32>(mapInteraction.viewSize.y);
    const F32 aspectRatio = (h > 0.0f) ? (w / h) : 1.0f;

    if (geoMapComponent) {
        JST_CHECK(geoMapComponent->updateUniforms({
            .centerLon = mapInteraction.centerLon,
            .centerLat = mapInteraction.centerLat,
            .zoom = mapInteraction.zoom,
            .aspectRatio = aspectRatio,
            .surfaceScale = mapInteraction.scale,
            .viewportWidth = w,
            .viewportHeight = h,
        }));
        JST_CHECK(geoMapComponent->present());
    }

    // Pack track segments from aircraft history.
    {
        std::lock_guard<std::mutex> lock(aircraftMutex);

        U64 segIdx = 0;
        const U64 maxSegments = maxAircraft * maxTrackPoints;

        for (const auto& [icao, ac] : aircraftMap) {
            if (!ac.hasPosition || ac.track.size() < 2) {
                continue;
            }
            for (U64 i = 0; i + 1 < ac.track.size() &&
                 segIdx < maxSegments; ++i, ++segIdx) {
                trackSegments[segIdx * 4 + 0] = static_cast<F32>(ac.track[i].second);      // lon1
                trackSegments[segIdx * 4 + 1] = static_cast<F32>(ac.track[i].first);       // lat1
                trackSegments[segIdx * 4 + 2] = static_cast<F32>(ac.track[i + 1].second);  // lon2
                trackSegments[segIdx * 4 + 3] = static_cast<F32>(ac.track[i + 1].first);   // lat2
            }
        }

        // Zero-fill remaining segments.
        for (U64 i = segIdx; i < maxSegments; ++i) {
            trackSegments[i * 4 + 0] = 0.0f;
            trackSegments[i * 4 + 1] = 0.0f;
            trackSegments[i * 4 + 2] = 0.0f;
            trackSegments[i * 4 + 3] = 0.0f;
        }
    }

    // Update track instance buffer.
    trackInstanceBuffer->update();

    // Update track uniforms.
    trackGpuUniforms.centerLon = mapInteraction.centerLon;
    trackGpuUniforms.centerLat = mapInteraction.centerLat;
    trackGpuUniforms.zoom = mapInteraction.zoom;
    trackGpuUniforms.aspectRatio = aspectRatio;
    trackGpuUniforms.surfaceScale = mapInteraction.scale;
    trackGpuUniforms.viewportWidth = w;
    trackGpuUniforms.viewportHeight = h;
    trackUniformBuffer->update();

    // Update aircraft uniforms.
    aircraftUniforms.centerLon = mapInteraction.centerLon;
    aircraftUniforms.centerLat = mapInteraction.centerLat;
    aircraftUniforms.zoom = mapInteraction.zoom;
    aircraftUniforms.aspectRatio = aspectRatio;
    aircraftUniforms.surfaceScale = mapInteraction.scale;
    aircraftUniforms.viewWidth = static_cast<int>(mapInteraction.viewSize.x);
    aircraftUniforms.viewHeight = static_cast<int>(mapInteraction.viewSize.y);

    // Update aircraft buffer.
    aircraftBuffer->update();

    const U64* countPtr = static_cast<const U64*>(aircraftCount.data());
    U64 countValue = countPtr[0];

    aircraftUniforms.aircraftCount = static_cast<int>(countValue);

    aircraftUniformBuffer->update();

    if (hoverText) {
        const F32 safeW = std::max(w, 1.0f);
        const F32 safeH = std::max(h, 1.0f);
        const Extent2D<F32> pixelSize = {
            (2.0f * mapInteraction.scale) / safeW,
            (2.0f * mapInteraction.scale) / safeH,
        };
        hoverText->updatePixelSize(pixelSize);

        std::string flightId;
        Extent2D<F32> flightNdc = {0.0f, 0.0f};
        {
            std::lock_guard<std::mutex> lock(aircraftMutex);
            flightId = hoveredFlightId;
            flightNdc = hoveredFlightNdc;
        }

        auto element = hoverText->get("flight");
        if (flightId.empty()) {
            element.fill = "";
            element.scale = 0.0f;
        } else {
            element.fill = flightId;
            element.scale = 0.9f;

            // Keep label readable near viewport borders by flipping anchor
            // side and clamping with a conservative width/height estimate.
            const F32 offsetX = 12.0f * pixelSize.x;
            const F32 offsetY = 16.0f * pixelSize.y;
            const F32 edgePadX = 8.0f * pixelSize.x;
            const F32 edgePadY = 8.0f * pixelSize.y;
            const bool placeLeft = (flightNdc.x > 0.65f);
            const bool placeBelow = (flightNdc.y > 0.75f);

            element.alignment.x = placeLeft ? 2 : 0;
            element.alignment.y = placeBelow ? 0 : 2;

            F32 labelX = flightNdc.x + (placeLeft ? -offsetX : offsetX);
            F32 labelY = flightNdc.y + (placeBelow ? -offsetY : offsetY);

            const F32 approxCharNdc = 7.0f * element.scale * pixelSize.x;
            const F32 approxWidthNdc =
                std::min(0.95f, approxCharNdc * static_cast<F32>(flightId.size()));
            const F32 approxHeightNdc = 16.0f * element.scale * pixelSize.y;

            if (element.alignment.x == 0) {
                labelX = std::clamp(labelX,
                                    -1.0f + edgePadX,
                                    1.0f - edgePadX - approxWidthNdc);
            } else {
                labelX = std::clamp(labelX,
                                    -1.0f + edgePadX + approxWidthNdc,
                                    1.0f - edgePadX);
            }

            labelY = std::clamp(labelY,
                                -1.0f + edgePadY + approxHeightNdc,
                                1.0f - edgePadY - approxHeightNdc);

            element.position = {labelX, labelY};
        }
        hoverText->update("flight", element);
        JST_CHECK(hoverText->present());
    }

    return Result::SUCCESS;
}

void AdsbImpl::clampMapView() {
    const F32 ar = static_cast<F32>(mapInteraction.viewSize.x) /
                   std::max(static_cast<F32>(mapInteraction.viewSize.y), 1.0f);

    // Minimum zoom so map fills the viewport on both axes.
    // Visible half-extent in Mercator: X = ar / (2*scale), Y = 1 / (2*scale).
    // Map Mercator range is [0,1] on both axes, so we need:
    //   ar / (2*scale) <= 0.5  →  scale >= ar
    //   1 / (2*scale) <= 0.5  →  scale >= 1
    const F32 minZoom = std::log2(std::max(ar, 1.0f));
    mapInteraction.zoom = std::clamp(mapInteraction.zoom, minZoom, 18.0f);

    const F32 scale = std::pow(2.0f, mapInteraction.zoom);

    // Clamp center in Mercator X so left/right edges stay in view.
    // Visible range: [cx - ar/(2*scale), cx + ar/(2*scale)] must be in [0,1].
    auto mercX = [](F32 lon) {
        return (lon + 180.0f) / 360.0f;
    };
    auto mercY = [&](F32 lat) {
        const F32 r = lat * static_cast<F32>(JST_PI) / 180.0f;
        return (1.0f - std::log(std::tan(r) + 1.0f / std::cos(r)) /
                       static_cast<F32>(JST_PI)) / 2.0f;
    };
    auto invMercY = [&](F32 my) {
        return std::atan(std::sinh(static_cast<F32>(JST_PI) * (1.0f - 2.0f * my))) *
               180.0f / static_cast<F32>(JST_PI);
    };

    const F32 halfExtentX = ar / (2.0f * scale);
    const F32 cx = std::clamp(mercX(mapInteraction.centerLon),
                              halfExtentX,
                              1.0f - halfExtentX);
    mapInteraction.centerLon = cx * 360.0f - 180.0f;

    // Clamp center in Mercator Y so top/bottom edges stay in view.
    const F32 halfExtentY = 1.0f / (2.0f * scale);
    const F32 cy = std::clamp(mercY(mapInteraction.centerLat), halfExtentY, 1.0f - halfExtentY);
    mapInteraction.centerLat = std::clamp(invMercY(cy), -85.051f, 85.051f);
}

void AdsbImpl::updateHoveredFlightFromCursor(const Extent2D<F32>& cursorPosNormalized) {
    const F32 viewW = std::max(static_cast<F32>(mapInteraction.viewSize.x), 1.0f);
    const F32 viewH = std::max(static_cast<F32>(mapInteraction.viewSize.y), 1.0f);
    const F32 aspectRatio = viewW / viewH;

    const F32 scale = std::pow(2.0f, mapInteraction.zoom);
    const F32 cursorNdcX = (cursorPosNormalized.x - 0.5f) * 2.0f;
    const F32 cursorNdcY = (0.5f - cursorPosNormalized.y) * 2.0f;

    auto mercX = [](F32 lon) {
        return (lon + 180.0f) / 360.0f;
    };
    auto mercY = [&](F32 lat) {
        const F32 r = lat * static_cast<F32>(JST_PI) / 180.0f;
        return (1.0f - std::log(std::tan(r) + 1.0f / std::cos(r)) /
                       static_cast<F32>(JST_PI)) / 2.0f;
    };

    const F32 cx = mercX(mapInteraction.centerLon);
    const F32 cy = mercY(mapInteraction.centerLat);

    // NDC-space hit radius avoids dependence on framebuffer resize timing.
    const F32 hoverRadiusNdc = 0.10f;
    const F32 hoverRadiusSq = hoverRadiusNdc * hoverRadiusNdc;

    F32 bestDistSq = std::numeric_limits<F32>::max();
    U32 bestIcao = 0;
    std::string bestCallsign;
    Extent2D<F32> bestNdc = {0.0f, 0.0f};
    bool found = false;

    std::lock_guard<std::mutex> lock(aircraftMutex);
    for (const auto& [icao, ac] : aircraftMap) {
        if (!ac.hasPosition) {
            continue;
        }

        const F32 acMx = mercX(static_cast<F32>(ac.longitude));
        const F32 acMy = mercY(static_cast<F32>(ac.latitude));

        const F32 acNdcX = (acMx - cx) * scale * 2.0f / aspectRatio;
        const F32 acNdcY = (cy - acMy) * scale * 2.0f;

        const F32 dx = (cursorNdcX - acNdcX);
        const F32 dy = (cursorNdcY - acNdcY);
        const F32 distSq = dx * dx + dy * dy;

        if (distSq <= hoverRadiusSq && distSq < bestDistSq) {
            bestDistSq = distSq;
            bestIcao = icao;
            bestCallsign = ac.hasCallsign ? ac.callsign : "";
            bestNdc = {acNdcX, acNdcY};
            found = true;
        }
    }

    if (!found) {
        hoveredFlightId.clear();
        hoveredFlightNdc = {0.0f, 0.0f};
        return;
    }

    const std::string icaoHex = jst::fmt::format("{:06X}", bestIcao);
    hoveredFlightId = bestCallsign.empty()
        ? jst::fmt::format("Flight {}", icaoHex)
        : jst::fmt::format("{} ({})", bestCallsign, icaoHex);
    hoveredFlightNdc = bestNdc;
}

void AdsbImpl::processMapInteraction(std::vector<SurfaceEvent>&& surfaceEvents,
                                     std::vector<MouseEvent>&& mouseEvents) {
    auto mercX = [](F32 lon) {
        return (lon + 180.0f) / 360.0f;
    };
    auto mercY = [&](F32 lat) {
        const F32 r = lat * static_cast<F32>(JST_PI) / 180.0f;
        return (1.0f - std::log(std::tan(r) + 1.0f / std::cos(r)) /
                       static_cast<F32>(JST_PI)) / 2.0f;
    };
    auto invMercY = [&](F32 my) {
        return std::atan(std::sinh(static_cast<F32>(JST_PI) * (1.0f - 2.0f * my))) *
               180.0f / static_cast<F32>(JST_PI);
    };

    SurfaceInteractionConfig interactionConfig;
    interactionConfig.enableZoom = false;
    interactionConfig.enablePan = false;
    interactionConfig.enableCursor = true;

    // Delegate generic event handling (resize/scale) to the helper.
    auto interactionMouseEvents = mouseEvents;
    surfaceInteraction = ProcessSurfaceInteraction(surfaceInteraction,
                                                   std::move(surfaceEvents),
                                                   std::move(interactionMouseEvents),
                                                   interactionConfig);

    if (surfaceInteraction.viewChanged) {
        mapInteraction.viewSize = surfaceInteraction.viewSize;
        mapInteraction.scale = surfaceInteraction.scale;
        mapInteraction.viewChanged = true;
    }

    if (surfaceInteraction.cursorMoved) {
        hasLastCursorSample = true;
        lastCursorSample = surfaceInteraction.cursorNormalized;
    }

    for (const auto& event : mouseEvents) {
        if (event.type != MouseEventType::Leave) {
            hasLastCursorSample = true;
            lastCursorSample = event.position;
        }

        switch (event.type) {
            case MouseEventType::Scroll: {
                const F32 ar = static_cast<F32>(mapInteraction.viewSize.x) /
                               std::max(static_cast<F32>(mapInteraction.viewSize.y), 1.0f);

                const F32 minZoom = std::log2(std::max(ar, 1.0f));
                const F32 oldZoom = mapInteraction.zoom;
                const F32 newZoom = std::clamp(oldZoom + event.scroll.y * 0.15f, minZoom, 18.0f);

                if (newZoom != oldZoom) {
                    // Convert cursor from normalized [0,1] to NDC [-1,1].
                    const F32 ndcX = (event.position.x - 0.5f) * 2.0f;
                    const F32 ndcY = (0.5f - event.position.y) * 2.0f;

                    const F32 oldScale = std::pow(2.0f, oldZoom);
                    const F32 newScale = std::pow(2.0f, newZoom);

                    // Shift center in Mercator space so the point
                    // under the cursor stays fixed.
                    const F32 factor = (1.0f / oldScale - 1.0f / newScale) / 2.0f;

                    const F32 newMercX = mercX(mapInteraction.centerLon) + ndcX * ar * factor;
                    const F32 newMercY = mercY(mapInteraction.centerLat) - ndcY * factor;

                    mapInteraction.centerLon = newMercX * 360.0f - 180.0f;
                    mapInteraction.centerLat = std::clamp(invMercY(newMercY), -85.051f, 85.051f);
                    mapInteraction.zoom = newZoom;
                }
                break;
            }
            case MouseEventType::Click: {
                if (event.button == MouseButton::Left) {
                    mapInteraction.dragging = true;
                    mapInteraction.dragAnchorX = event.position.x;
                    mapInteraction.dragAnchorY = event.position.y;
                    mapInteraction.dragStartLon = mapInteraction.centerLon;
                    mapInteraction.dragStartLat = mapInteraction.centerLat;
                }
                break;
            }
            case MouseEventType::Release: {
                mapInteraction.dragging = false;
                break;
            }
            case MouseEventType::Move: {
                if (mapInteraction.dragging) {
                    const F32 dx = event.position.x - mapInteraction.dragAnchorX;
                    const F32 dy = event.position.y - mapInteraction.dragAnchorY;
                    const F32 ar = static_cast<F32>(mapInteraction.viewSize.x) /
                                   std::max(static_cast<F32>(mapInteraction.viewSize.y), 1.0f);

                    // Pan in Mercator space: convert delta pixels
                    // to Mercator units then back to lon/lat.
                    const F32 scale = std::pow(2.0f, mapInteraction.zoom);
                    const F32 dMercX = dx / scale * ar;
                    const F32 dMercY = dy / scale;

                    mapInteraction.centerLon = mapInteraction.dragStartLon - dMercX * 360.0f;

                    // Inverse Mercator for latitude.
                    const F32 centerMercY = mercY(mapInteraction.dragStartLat);
                    const F32 newMercY = centerMercY - dMercY;
                    mapInteraction.centerLat = invMercY(newMercY);
                    mapInteraction.centerLat = std::clamp(mapInteraction.centerLat, -85.051f, 85.051f);
                }
                break;
            }
            case MouseEventType::Leave: {
                mapInteraction.dragging = false;
                hasLastCursorSample = false;
                std::lock_guard<std::mutex> lock(aircraftMutex);
                hoveredFlightId.clear();
                hoveredFlightNdc = {0.0f, 0.0f};
                break;
            }
            default:
                break;
        }
    }

    clampMapView();

    if (!hasLastCursorSample) {
        std::lock_guard<std::mutex> lock(aircraftMutex);
        hoveredFlightId.clear();
        hoveredFlightNdc = {0.0f, 0.0f};
        return;
    }

    updateHoveredFlightFromCursor(lastCursorSample);
}

}  // namespace Jetstream::Modules
