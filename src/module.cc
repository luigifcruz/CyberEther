#include "jetstream/types.hh"
#include <jetstream/runtime.hh>
#include <jetstream/logger.hh>
#include <jetstream/module.hh>
#include <jetstream/detail/module_impl.hh>

#ifdef JST_OS_BROWSER
#include <utility>
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#endif

namespace Jetstream {

Module::Module(const DeviceType& device,
               const RuntimeType& runtime,
               const ProviderType& provider,
               const std::shared_ptr<Module::Impl>& impl,
               const std::shared_ptr<Module::Context>& context,
               const std::shared_ptr<Module::Config>& stagedConfig,
               const std::shared_ptr<Module::Config>& candidateConfig) : impl(impl){
    impl->_device = device;
    impl->_runtime = runtime;
    impl->_provider = provider;
    impl->_context = context;
    impl->_stagedConfig = stagedConfig;
    impl->_candidateConfig = candidateConfig;
}

Result Module::create(const std::string& name,
                      const Config& config,
                      const TensorMap& inputs,
                      const std::shared_ptr<Render::Window>& render) {
    Parser::Map serializedConfig;
    JST_CHECK(config.serialize(serializedConfig));
    return Module::create(name,
                          serializedConfig,
                          inputs,
                          render);
}

Result Module::create(const std::string& name,
                      const Parser::Map& config,
                      const TensorMap& inputs,
                      const std::shared_ptr<Render::Window>& render) {
    // Set implementation variables.

    impl->_inputs = inputs;
    impl->_outputs = TensorMap();
    impl->_interface = std::make_shared<Interface>();
    impl->_surface = std::make_shared<Surface>();
    impl->_name = name;
    impl->_render = render;

    JST_DEBUG("[MODULE] Creating module '{}'.", impl->_name);

    // Validate configuration.

    JST_CHECK(impl->_candidateConfig->deserialize(config));
    JST_CHECK(impl->validate());

    // Commit candidate.

    JST_CHECK(impl->_stagedConfig->deserialize(config));

    // Define module interface.

    JST_CHECK(impl->define());

    // Verify module taints.

    bool taintDiscontiguous = false;

    if (impl->_taint != Module::Taint::CLEAN) {
        JST_TRACE("[MODULE] Module ('{}') is tainted. Verifying...", impl->_name);

        if ((impl->_taint & Taint::DISCONTIGUOUS) == Taint::DISCONTIGUOUS) {
            taintDiscontiguous = true;
        }
    }

    // Check if module provides all requested inputs.

    for (const auto& key : impl->_interface->inputs()) {
        if (!impl->_inputs.contains(key)) {
            JST_ERROR("[MODULE] Module '{}' requested missing input '{}'.", impl->_name, key);
            return Result::ERROR;
        }
    }

    // Verify input tensors device matches module device.

    for (const auto& [name, link] : inputs) {
        if (link.tensor.device() != impl->_device) {
            JST_ERROR("[MODULE] Input tensor device ('{}', DeviceType::{})"
                      " doesn't match the module device ('{}', DeviceType::{}).",
                      name, link.tensor.device(), impl->_name, impl->_device);
            return Result::ERROR;
        }

        if (!link.tensor.validShape()) {
            JST_ERROR("[MODULE] Input tensor ('{}') is invalid.", name);
            return Result::ERROR;
        }

        if (link.tensor.size() == 0) {
            JST_ERROR("[MODULE] Module ('{}') input tensor ('{}') size is zero.", impl->_name, name);
            return Result::ERROR;
        }

        if (!link.tensor.contiguous() && !taintDiscontiguous) {
            JST_ERROR("[MODULE] Contiguous tensor expected for module ('{}') input tensor ('{}').", impl->_name, name);
            return Result::ERROR;
        }
    }

    // Creating module.

#ifdef JST_OS_BROWSER
    if ((impl->_taint & Taint::BROWSER_MAIN_THREAD) == Taint::BROWSER_MAIN_THREAD) {
        Result result;
        std::pair<Impl*, Result*> ctx{impl.get(), &result};
        emscripten_proxy_sync(
            emscripten_proxy_get_system_queue(),
            emscripten_main_runtime_thread_id(),
            Impl::proxyCreate,
            &ctx);
        JST_CHECK(result);
    } else {
        JST_CHECK(impl->create());
    }
#else
    JST_CHECK(impl->create());
#endif

    // Check if module provides all requested outputs.

    for (const auto& key : impl->_interface->outputs()) {
        if (!impl->_outputs.contains(key)) {
            JST_ERROR("[MODULE] Module '{}' didn't create an expected output '{}'.", impl->_name, key);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result Module::destroy() {
#ifdef JST_OS_BROWSER
    if ((impl->_taint & Taint::BROWSER_MAIN_THREAD) == Taint::BROWSER_MAIN_THREAD) {
        Result result;
        std::pair<Impl*, Result*> ctx{impl.get(), &result};
        emscripten_proxy_sync(
            emscripten_proxy_get_system_queue(),
            emscripten_main_runtime_thread_id(),
            Impl::proxyDestroy,
            &ctx);
        JST_CHECK(result);
    } else {
        JST_CHECK(impl->destroy());
    }
#else
    JST_CHECK(impl->destroy());
#endif

    return Result::SUCCESS;
}

Result Module::reconfigure(const Parser::Map& config, const bool& validateOnly) {
    // Deserialize new configuration.

    JST_CHECK(impl->_candidateConfig->deserialize(config));

    // Return early if the configuration is unchanged.

    if (impl->_candidateConfig->hash() == impl->_stagedConfig->hash()) {
        return Result::SUCCESS;
    }

    // Validate configuration and reconfigure the module if something changed.

    JST_CHECK(impl->validate());
    if (!validateOnly) {
        JST_CHECK(impl->reconfigure());
    }

    return Result::SUCCESS;
}

Result Module::config(Parser::Map& config) const {
    return impl->_stagedConfig->serialize(config);
}

const std::shared_ptr<Module::Context>& Module::context() {
    return impl->_context;
}

const Module::Config& Module::config() const {
    return *impl->_stagedConfig;
}

const Module::Taint& Module::taint() const {
    return impl->_taint;
}

const TensorMap& Module::inputs() const {
    return impl->_inputs;
}

const TensorMap& Module::outputs() const {
    return impl->_outputs;
}

const std::shared_ptr<Module::Interface>& Module::interface() const {
    return impl->_interface;
}

const std::string& Module::name() const {
    return impl->_name;
}

const DeviceType& Module::device() const {
    return impl->_device;
}

const RuntimeType& Module::runtime() const {
    return impl->_runtime;
}

const ProviderType& Module::provider() const {
    return impl->_provider;
}

const std::shared_ptr<Module::Surface>& Module::surface() {
    return impl->_surface;
}

}  // namespace Jetstream
