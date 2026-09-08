#include "jetstream/render/components/map_layer.hh"

#include <algorithm>
#include <exception>

namespace Jetstream::Render::Components {

Result MapLayerStack::add(const std::shared_ptr<MapLayer>& layer) {
    if (active || !layer ||
        std::find(layers.begin(), layers.end(), layer) != layers.end()) {
        JST_ERROR("[GEOMAP] Layers must be unique and attached before creation.");
        return Result::ERROR;
    }
    layers.push_back(layer);
    return Result::SUCCESS;
}

Result MapLayerStack::create(Window* window, const MapContext& context) {
    if (active) return Result::ERROR;
    active = true;
    for (const auto& layer : layers) {
        // Include the current layer in rollback even if creation only partially
        // succeeds. Also preserve cleanup responsibility if a hook throws.
        ++createdCount;
        try {
            const Result result = layer->create(window, context);
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                destroy(window);
                return result;
            }
        } catch (...) {
            destroy(window);
            throw;
        }
    }
    return Result::SUCCESS;
}

Result MapLayerStack::destroy(Window* window) {
    captured.reset();
    Result result = Result::SUCCESS;
    std::exception_ptr exception;
    while (createdCount > 0) {
        try {
            const Result cleanup = layers[--createdCount]->destroy(window);
            if (cleanup != Result::SUCCESS && cleanup != Result::RELOAD) {
                result = cleanup;
            }
        } catch (...) {
            if (!exception) exception = std::current_exception();
        }
    }
    active = false;
    if (exception) std::rethrow_exception(exception);
    return result;
}

Result MapLayerStack::surface(Render::Surface::Config& config) {
    if (!active) return Result::ERROR;
    // Do not leave half an assembled surface in the caller on failure.
    auto next = config;
    for (const auto& layer : layers) {
        JST_CHECK(layer->surface(next));
    }
    config = std::move(next);
    return Result::SUCCESS;
}

Result MapLayerStack::present(const MapContext& context) {
    if (!active) return Result::ERROR;
    for (const auto& layer : layers) {
        JST_CHECK(layer->present(context));
    }
    return Result::SUCCESS;
}

bool MapLayerStack::onEvent(const MouseEvent& event, const MapContext& context) {
    if (!active) return false;
    // Everyone must clear hover state on leave, regardless of stacking order.
    if (event.type == MouseEventType::Leave) {
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            (*it)->onEvent(event, context);
        }
        captured.reset();
        return false;
    }
    if (captured) {
        captured->onEvent(event, context);
        if (event.type == MouseEventType::Release) captured.reset();
        return true;
    }
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        const auto result = (*it)->onEvent(event, context);
        if (result == MapLayer::EventResult::Ignored) continue;
        if (result == MapLayer::EventResult::Capture &&
            event.type == MouseEventType::Click) {
            captured = *it;
        }
        return true;
    }
    return false;
}

}  // namespace Jetstream::Render::Components
