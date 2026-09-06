#include <jetstream/detail/module_impl.hh>
#include <jetstream/detail/module_context_impl.hh>
#include <jetstream/detail/module_interface_impl.hh>
#include <jetstream/detail/module_surface_impl.hh>

#include <mutex>

#ifdef JST_OS_BROWSER
#include <utility>
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#endif

namespace Jetstream {

#ifdef JST_OS_BROWSER
void Module::Impl::proxyCreate(void* arg) {
    auto* ctx = static_cast<std::pair<Impl*, Result*>*>(arg);
    *ctx->second = ctx->first->create();
}

void Module::Impl::proxyDestroy(void* arg) {
    auto* ctx = static_cast<std::pair<Impl*, Result*>*>(arg);
    *ctx->second = ctx->first->destroy();
}
#endif

Result Module::Impl::destroyImplementation() {
#ifdef JST_OS_BROWSER
    if ((_taint & Taint::BROWSER_MAIN_THREAD) == Taint::BROWSER_MAIN_THREAD) {
        Result result;
        std::pair<Impl*, Result*> ctx{this, &result};
        emscripten_proxy_sync(
            emscripten_proxy_get_system_queue(),
            emscripten_main_runtime_thread_id(),
            Impl::proxyDestroy,
            &ctx);
        return result;
    }
#endif

    return destroy();
}

Result Module::Impl::validate() {
    return Result::SUCCESS;
}

Result Module::Impl::define() {
    return Result::SUCCESS;
}

Result Module::Impl::create() {
    return Result::SUCCESS;
}

Result Module::Impl::destroy() {
    return Result::SUCCESS;
}

Result Module::Impl::reconfigure() {
    return Result::RECREATE;
}

Result Module::Impl::requestConfigChange(const Parser::Map& config) {
    std::lock_guard lock(_configChangeMutex);
    if (!_configChangesPending) {
        return Result::ERROR;
    }
    // Reject the whole patch if any field has not been explicitly bound.
    for (const auto& entry : config) {
        if (!_configChangeBindings.contains(entry.key)) {
            JST_ERROR("[MODULE] Configuration edit '{}' is not bound for '{}'.",
                      entry.key, _name);
            return Result::ERROR;
        }
    }
    for (const auto& entry : config) {
        _pendingConfigChanges[_configChangeBindings.at(entry.key)] = entry.value;
    }
    if (!config.empty()) {
        _configChangesPending->store(true);
    }
    return Result::SUCCESS;
}

bool Module::Impl::configChangeEnabled(const std::string& key) const {
    std::lock_guard lock(_configChangeMutex);
    return _configChangesPending && _configChangeBindings.contains(key);
}

bool Module::Impl::configChangePending() const {
    std::lock_guard lock(_configChangeMutex);
    return _configChangeInFlight || !_pendingConfigChanges.empty();
}

Result Module::Impl::configChangeResult() const {
    std::lock_guard lock(_configChangeMutex);
    return _configChangeResult;
}

void Module::Impl::invalidateConfigChanges() {
    std::lock_guard lock(_configChangeMutex);
    _configChangeBindings.clear();
    _pendingConfigChanges.clear();
    _configChangeInFlight = false;
    _configChangeResult = Result::ERROR;
    // Disconnect this module without clearing notifications from other modules.
    _configChangesPending.reset();
}

Result Module::Impl::defineTaint(const Taint& taint) {
    _taint = taint | _taint;

    return Result::SUCCESS;
}

Result Module::Impl::defineInterfaceInput(const std::string& key) {
    for (const auto& input : _interface->impl->inputs) {
        if (input == key) {
            JST_ERROR("[MODULE] Input '{}' already exists", key);
            return Result::ERROR;
        }
    }
    _interface->impl->inputs.push_back(key);
    return Result::SUCCESS;
}

Result Module::Impl::defineInterfaceOutput(const std::string& key) {
    for (const auto& output : _interface->impl->outputs) {
        if (output == key) {
            JST_ERROR("[MODULE] Output '{}' already exists", key);
            return Result::ERROR;
        }
    }
    _interface->impl->outputs.push_back(key);
    return Result::SUCCESS;
}

const TensorMap& Module::Impl::inputs() const {
    return _inputs;
}

TensorMap& Module::Impl::outputs() {
    return _outputs;
}

const std::string& Module::Impl::name() const {
    return _name;
}

const DeviceType& Module::Impl::device() const {
    return _device;
}

const RuntimeType& Module::Impl::runtime() const {
    return _runtime;
}

const ProviderType& Module::Impl::provider() const {
    return _provider;
}

const Module::Taint& Module::Impl::taint() const {
    return _taint;
}

const std::shared_ptr<Render::Window>& Module::Impl::render() {
    return _render;
}

const std::shared_ptr<Flowgraph::Environment>& Module::Impl::environment() {
    return _context->environment();
}

const std::shared_ptr<Flowgraph::Environment>& Module::Impl::environment() const {
    return _context->environment();
}

const std::shared_ptr<Flowgraph::View>& Module::Impl::view() {
    return _context->view();
}

const std::shared_ptr<Flowgraph::View>& Module::Impl::view() const {
    return _context->view();
}

Result Module::Impl::surfaceCreateManifest(SurfaceManifest&& manifest) {
    std::lock_guard<std::mutex> lock(_surface->impl->manifestMutex);
    _surface->impl->manifests.push_back(std::move(manifest));
    return Result::SUCCESS;
}

Result Module::Impl::surfaceUpdateManifestSize(const std::string& id, const Extent2D<U64>& size) {
    std::lock_guard<std::mutex> lock(_surface->impl->manifestMutex);
    for (auto& manifest : _surface->impl->manifests) {
        if (manifest.id == id) {
            manifest.size = size;
            return Result::SUCCESS;
        }
    }
    return Result::ERROR;
}

std::vector<MouseEvent> Module::Impl::surfaceConsumeMouseEvents() {
    std::lock_guard<std::mutex> lock(_surface->impl->eventMutex);
    return _surface->impl->eventBuffer.consumeMouseEvents();
}

std::vector<SurfaceEvent> Module::Impl::surfaceConsumeSurfaceEvents() {
    std::lock_guard<std::mutex> lock(_surface->impl->eventMutex);
    return _surface->impl->eventBuffer.consumeSurfaceEvents();
}

}  // namespace Jetstream
