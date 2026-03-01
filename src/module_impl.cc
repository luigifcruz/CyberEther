#include <jetstream/detail/module_impl.hh>
#include <jetstream/detail/module_context_impl.hh>
#include <jetstream/detail/module_interface_impl.hh>
#include <jetstream/detail/module_surface_impl.hh>

#ifdef JST_OS_BROWSER
#include <utility>
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

std::shared_ptr<Render::Window>& Module::Impl::render() {
    return _render;
}

Result Module::Impl::surfaceCreateManifest(SurfaceManifest&& manifest) {
    _surface->impl->manifests.push_back(std::move(manifest));
    return Result::SUCCESS;
}

Result Module::Impl::surfaceUpdateManifestSize(const std::string& id, const Extent2D<U64>& size) {
    for (auto& manifest : _surface->impl->manifests) {
        if (manifest.id == id) {
            manifest.size = size;
            return Result::SUCCESS;
        }
    }
    return Result::ERROR;
}

std::vector<MouseEvent> Module::Impl::surfaceConsumeMouseEvents() {
    return _surface->impl->eventBuffer.consumeMouseEvents();
}

std::vector<SurfaceEvent> Module::Impl::surfaceConsumeSurfaceEvents() {
    return _surface->impl->eventBuffer.consumeSurfaceEvents();
}

}  // namespace Jetstream
