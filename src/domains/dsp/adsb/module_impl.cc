#include "module_impl.hh"

#include <any>
#include <chrono>

namespace Jetstream::Modules {

Result AdsbImpl::validate() {
    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.hasAttribute("frequency")) {
        const std::any value = inputTensor.attribute("frequency");
        if (!std::any_cast<F32>(&value)) {
            JST_ERROR("[MODULE_ADSB] Input frequency metadata must have type F32.");
            return Result::ERROR;
        }
    }

    if (inputTensor.hasAttribute("sampleRate")) {
        const std::any value = inputTensor.attribute("sampleRate");
        if (!std::any_cast<F32>(&value)) {
            JST_ERROR("[MODULE_ADSB] Input sample rate metadata must have type F32.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result AdsbImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::SURFACE));
    JST_CHECK(defineInterfaceInput("signal"));
    return Result::SUCCESS;
}

Result AdsbImpl::create() {
    input = inputs().at("signal").tensor;

    if (input.hasAttribute("frequency")) {
        const std::any value = input.attribute("frequency");
        if (const auto* frequency = std::any_cast<F32>(&value)) {
            JST_INFO("[MODULE_ADSB] Input frequency: {:.2f} MHz", *frequency / 1e6f);
        }
    }

    if (input.hasAttribute("sampleRate")) {
        const std::any value = input.attribute("sampleRate");
        if (const auto* sampleRate = std::any_cast<F32>(&value)) {
            JST_INFO("[MODULE_ADSB] Input sample rate: {:.2f} MHz", *sampleRate / 1e6f);
        }
    }

    {
        std::lock_guard<std::mutex> lock(aircraftMutex);
        aircraftMap.clear();
        aircraftTableDirty = true;
    }
    updateAircraftTable();
    return Result::SUCCESS;
}

std::string AdsbImpl::getAircraftTable() const {
    return aircraftTable.get();
}

void AdsbImpl::updateAircraftTable() {
    std::string nextAircraftTable;
    {
        std::lock_guard<std::mutex> lock(aircraftMutex);
        if (!aircraftTableDirty) return;

        if (aircraftMap.empty()) {
            nextAircraftTable = "No aircraft detected.";
        } else {
            nextAircraftTable += "ICAO\tCallsign\tAlt (ft)\tSpeed (kt)\tHdg\tLat\tLon\n";
            for (const auto& [icao, ac] : aircraftMap) {
                nextAircraftTable += jst::fmt::format("{:06X}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    icao,
                    ac.hasCallsign ? ac.callsign : "-",
                    ac.hasAltitude ? jst::fmt::format("{}", ac.altitude) : "-",
                    ac.hasVelocity ? jst::fmt::format("{:.0f}", ac.speed) : "-",
                    ac.hasVelocity ? jst::fmt::format("{:.0f}", ac.heading) : "-",
                    ac.hasPosition ? jst::fmt::format("{:.4f}", ac.latitude) : "-",
                    ac.hasPosition ? jst::fmt::format("{:.4f}", ac.longitude) : "-");
            }
        }
        aircraftTableDirty = false;
    }
    aircraftTable.publish(std::move(nextAircraftTable));
}

Result AdsbImpl::destroy() {
    return destroyPresent();
}

Result AdsbImpl::createPresent() {
    auto& window = render();
    if (!window) return Result::SUCCESS;

    mapState = std::make_shared<AdsbMapState>();
    geoMapComponent = std::make_unique<Render::Components::GeoMap>(
        Render::Components::GeoMap::Config{});
    // This is the entire domain-specific contribution to the base map. Layer
    // implementations and their shaders live alongside this module.
    JST_CHECK(AddAdsbMapLayers(*geoMapComponent, mapState));
    if (savedMapView) JST_CHECK(geoMapComponent->updateUniforms(*savedMapView));
    const SurfaceEvent initialView{SurfaceEventType::Resize, viewSize, viewScale};
    JST_CHECK(geoMapComponent->processInteraction({&initialView, 1}, {}));
    JST_CHECK(geoMapComponent->create(window.get()));

    {
        Render::Texture::Config cfg;
        cfg.size = viewSize;
        JST_CHECK(window->build(framebufferTexture, cfg));
    }
    {
        Render::Surface::Config cfg;
        cfg.framebuffer = framebufferTexture;
        cfg.multisampled = false;
        JST_CHECK(geoMapComponent->surface(cfg));
        JST_CHECK(window->build(renderSurface, cfg));
        JST_CHECK(window->bind(renderSurface));
    }
    return surfaceCreateManifest({
        .id = "default",
        .size = viewSize,
        .surface = framebufferTexture,
    });
}

Result AdsbImpl::destroyPresent() {
    auto& window = render();
    if (!window) return Result::SUCCESS;
    // The surface must release its programs before layer resources are torn
    // down. This also handles a partially completed createPresent().
    if (renderSurface) JST_CHECK(window->unbind(renderSurface));
    if (geoMapComponent) {
        savedMapView = geoMapComponent->getUniforms();
        JST_CHECK(geoMapComponent->destroy(window.get()));
    }
    renderSurface.reset();
    framebufferTexture.reset();
    geoMapComponent.reset();
    mapState.reset();
    return Result::SUCCESS;
}

Result AdsbImpl::present() {
    if (!renderSurface || !geoMapComponent) return Result::SUCCESS;

    // One coherent presentation snapshot: icon positions, tracks and picking
    // no longer read independently from compute-owned tensors and maps.
    mapState->aircraft.clear();
    const auto now = static_cast<U64>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    {
        std::lock_guard<std::mutex> lock(aircraftMutex);
        for (const auto& [icao, ac] : aircraftMap) {
            if (!ac.hasPosition) continue;
            mapState->aircraft.push_back({
                .icao = icao,
                .callsign = ac.hasCallsign ? ac.callsign : "",
                .latitude = static_cast<F32>(ac.latitude),
                .longitude = static_cast<F32>(ac.longitude),
                .heading = ac.heading,
                .altitude = static_cast<F32>(ac.altitude),
                .groundSpeed = ac.speed,
                .hasAltitude = ac.hasAltitude,
                .hasVelocity = ac.hasVelocity,
                .positionAgeSeconds = ac.lastPositionTimestamp > 0
                    ? std::optional<F32>(static_cast<F32>(now >= ac.lastPositionTimestamp
                        ? now - ac.lastPositionTimestamp : 0) / 1000.0f)
                    : std::nullopt,
                .track = ac.track,
            });
        }
    }

    const auto surfaceEvents = surfaceConsumeSurfaceEvents();
    const auto mouseEvents = surfaceConsumeMouseEvents();
    JST_CHECK(geoMapComponent->processInteraction(surfaceEvents, mouseEvents));
    if (const auto fitted = mapAutoFit.update(mapState->aircraft,
                                              geoMapComponent->getContext(), mouseEvents, now)) {
        JST_CHECK(geoMapComponent->updateUniforms(*fitted));
    }
    const auto& view = geoMapComponent->getUniforms();
    const Extent2D<U64> nextSize = {
        static_cast<U64>(view.viewportWidth), static_cast<U64>(view.viewportHeight),
    };
    if (nextSize.x != viewSize.x || nextSize.y != viewSize.y) {
        viewSize = nextSize;
        renderSurface->size(viewSize);
        JST_CHECK(surfaceUpdateManifestSize("default", viewSize));
    }
    viewScale = view.surfaceScale;
    return geoMapComponent->present();
}

}  // namespace Jetstream::Modules
