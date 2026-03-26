#include <jetstream/block.hh>
#include <jetstream/detail/block_impl.hh>
#include <jetstream/module_surface.hh>
#include <jetstream/detail/block_interface_impl.hh>

namespace Jetstream {

Result Block::Impl::validate() {
    return Result::SUCCESS;
}

Result Block::Impl::configure() {
    return Result::SUCCESS;
}

Result Block::Impl::define() {
    return Result::SUCCESS;
}

Result Block::Impl::create() {
    return Result::SUCCESS;
}

Result Block::Impl::destroy() {
    return Result::SUCCESS;
}

Result Block::Impl::moduleCreate(const std::string name,
                                 const std::shared_ptr<Module::Config>& config,
                                 const TensorMap& inputs) {
    JST_DEBUG("[BLOCK] Creating module '{}' of type '{}' inside block '{}'.",
              name, config->type(), _name);

    // Check if module exists.

    if (_modules.contains(name)) {
        JST_ERROR("[BLOCK] Module '{}' already exists inside block '{}'.",
                  name, _name);
        return Result::ERROR;
    }

    // Build module from registry.

    std::shared_ptr<Module> module;
    JST_CHECK(Registry::BuildModule(config->type(),
                                    _device,
                                    _runtime,
                                    _provider,
                                    module));
    _modules[name] = {module, config};
    _moduleOrder.push_back(name);

    // Clone input tensors to give each module an independent layout.

    TensorMap clonedInputs;
    for (const auto& [key, link] : inputs) {
        clonedInputs[key] = {link.block, link.port, link.tensor.clone()};
    }

    // Create module.

    const auto& blockModuleName = jst::fmt::format("{}-{}", _name, name);
    JST_CHECK(module->create(blockModuleName, *config, clonedInputs, _render));

    // Add module to scheduler.

    JST_CHECK(_scheduler->add(module));

    // Cache surface providers.

    if ((module->taint() & Module::Taint::SURFACE) != Module::Taint::CLEAN) {
        _surfaces.push_back(module->surface());
    }

    return Result::SUCCESS;
};

Result Block::Impl::moduleDestroy(const std::string name) {
    JST_DEBUG("[BLOCK] Destroying module '{}' inside block '{}'.", name, _name);

    // Check if module exists.

    if (!_modules.contains(name)) {
        JST_ERROR("[BLOCK] Module '{}' doesn't exist inside block '{}'.", name, _name);
        return Result::ERROR;
    }

    // Check if module is last in order (can only destroy in reverse order).

    if (_moduleOrder.empty() || _moduleOrder.back() != name) {
        JST_ERROR("[BLOCK] Module '{}' must be destroyed in reverse creation order.", name);
        return Result::ERROR;
    }
    _moduleOrder.pop_back();

    // Get module and remove from list.

    auto entry = std::move(_modules[name]);
    _modules.erase(name);

    // Remove module from scheduler.

    JST_CHECK(_scheduler->remove(entry.module));

    // Remove from surface providers cache.

    _surfaces.erase(std::remove(_surfaces.begin(), _surfaces.end(), entry.module->surface()),
                    _surfaces.end());

    // Destroy module.

    JST_CHECK(entry.module->destroy());

    return Result::SUCCESS;
}

Result Block::Impl::moduleReconfigure(const std::string name, const bool& validateOnly) {
    // Check if module exists.

    if (!_modules.contains(name)) {
        JST_ERROR("[BLOCK] Module '{}' doesn't exist inside block '{}'.", name, _name);
        return Result::ERROR;
    }
    const auto& entry = _modules.at(name);

    // Update module configuration.

    Parser::Map candidate;
    JST_CHECK(entry.config->serialize(candidate));
    JST_CHECK(entry.module->reconfigure(candidate, validateOnly));

    return Result::SUCCESS;
}

std::shared_ptr<Module> Block::Impl::moduleHandle(const std::string& name) {
    if (!_modules.contains(name)) {
        return nullptr;
    }
    return _modules.at(name).module;
}

Result Block::Impl::moduleExposeOutput(const std::string blockPort,
                                       const std::pair<std::string, std::string>& moduleOutput) {
    const auto& [moduleName, modulePort] = moduleOutput;

    JST_DEBUG("[BLOCK] Exposing module output '{}.{}' as block output '{}.{}'.",
              moduleName, modulePort, _name, blockPort);

    // Check if module exists.

    if (!_modules.contains(moduleName)) {
        JST_ERROR("[BLOCK] Module '{}' doesn't exist inside block '{}'.", moduleName, _name);
        return Result::ERROR;
    }
    const auto& entry = _modules.at(moduleName);

    // Check if module port exists.

    if (!entry.module->outputs().contains(modulePort)) {
        JST_ERROR("[BLOCK] Port '{}' doesn't exist inside module '{}'.", modulePort, moduleName);
        return Result::ERROR;
    }

    // Check if block port exists.

    if (_outputs.contains(blockPort)) {
        JST_ERROR("[BLOCK] Block port '{}' already exists inside block '{}'.", blockPort, _name);
        return Result::ERROR;
    }

    // Store tensor reference.

    const auto& moduleTensor = entry.module->outputs().at(modulePort);
    _outputs[blockPort] = {_name, blockPort, moduleTensor.tensor};

    return Result::SUCCESS;
}

TensorLink Block::Impl::moduleGetOutput(const std::pair<std::string, std::string>& moduleOutput) {
    const auto& [moduleName, modulePort] = moduleOutput;

    // Check if module exists.

    if (!_modules.contains(moduleName)) {
        JST_ERROR("[BLOCK] Module '{}' doesn't exist inside block '{}'.", moduleName, _name);
        return {};
    }
    const auto& entry = _modules.at(moduleName);

    // Check if module port exists.

    if (!entry.module->outputs().contains(modulePort)) {
        JST_ERROR("[BLOCK] Port '{}' doesn't exist inside module '{}'.", modulePort, moduleName);
        return {};
    }

    return entry.module->outputs().at(modulePort);
}

const TensorMap& Block::Impl::inputs() const {
    return _inputs;
}

TensorMap& Block::Impl::outputs() {
    return _outputs;
}

const std::string& Block::Impl::name() const {
    return _name;
}

const DeviceType& Block::Impl::device() const {
    return _device;
}

const RuntimeType& Block::Impl::runtime() const {
    return _runtime;
}

const ProviderType& Block::Impl::provider() const {
    return _provider;
}

std::shared_ptr<Instance>& Block::Impl::instance() {
    return _instance;
}

std::shared_ptr<Render::Window>& Block::Impl::render() {
    return _render;
}

std::shared_ptr<Scheduler>& Block::Impl::scheduler() {
    return _scheduler;
}

const std::vector<std::shared_ptr<Module::Surface>>& Block::Impl::surfaces() const {
    return _surfaces;
}

const std::vector<std::string>& Block::Impl::modules() const {
    return _moduleOrder;
}

Result Block::Impl::defineInterfaceInput(const std::string& key,
                                         const std::string& label,
                                         const std::string& help) {
    for (const auto& input : _interface->impl->inputs) {
        if (input.first == key) {
            JST_ERROR("[BLOCK] Input '{}' already exists", key);
            return Result::ERROR;
        }
    }
    _interface->impl->inputs.push_back({key, {label, "", help, {}}});
    return Result::SUCCESS;
}

Result Block::Impl::defineInterfaceOutput(const std::string& key,
                                          const std::string& label,
                                          const std::string& help) {
    for (const auto& output : _interface->impl->outputs) {
        if (output.first == key) {
            JST_ERROR("[BLOCK] Output '{}' already exists", key);
            return Result::ERROR;
        }
    }
    _interface->impl->outputs.push_back({key, {label, "", help, {}}});
    return Result::SUCCESS;
}

Result Block::Impl::defineInterfaceConfig(const std::string& key,
                                          const std::string& label,
                                          const std::string& help,
                                          const std::string& format) {
    for (const auto& config : _interface->impl->configs) {
        if (config.first == key) {
            JST_ERROR("[BLOCK] Config '{}' already exists", key);
            return Result::ERROR;
        }
    }
    _interface->impl->configs.push_back({key, {label, format, help, {}}});
    return Result::SUCCESS;
}

Result Block::Impl::defineInterfaceMetric(const std::string& key,
                                          const std::string& label,
                                          const std::string& help,
                                          const std::string& format,
                                          std::function<std::any()> metric) {
    for (const auto& m : _interface->impl->metrics) {
        if (m.first == key) {
            JST_ERROR("[BLOCK] Metric '{}' already exists", key);
            return Result::ERROR;
        }
    }
    _interface->impl->metrics.push_back({key, {label, format, help, std::move(metric)}});
    return Result::SUCCESS;
}

}  // namespace Jetstream
