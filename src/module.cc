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
    if (impl->_state != State::NONE && impl->_state != State::DESTROYED) {
        JST_ERROR("[MODULE] Cannot create module '{}' in its current lifecycle state.", name);
        return Result::ERROR;
    }

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
    if (impl->_state != State::NONE && impl->_state != State::DESTROYED) {
        JST_ERROR("[MODULE] Cannot create module '{}' in its current lifecycle state.", name);
        return Result::ERROR;
    }

    impl->_state = State::CREATING;

    // Set implementation variables.

    impl->_inputs = inputs;
    impl->_outputs = TensorMap();
    impl->_interface = std::make_shared<Interface>();
    impl->_surface = std::make_shared<Surface>();
    impl->_name = name;
    impl->_render = render;

    JST_DEBUG("[MODULE] Creating module '{}'.", impl->_name);

    const auto stopCreating = [&](const Result result) {
        impl->_state = (result == Result::INCOMPLETE) ? State::INCOMPLETE : State::ERRORED;
        return result;
    };

    // Validate configuration.

    {
        const auto result = impl->_candidateConfig->deserialize(config);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return stopCreating(result);
        }
    }

    {
        const auto result = impl->validate();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return stopCreating(result);
        }
    }

    // Commit candidate.

    {
        const auto result = impl->_stagedConfig->deserialize(config);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return stopCreating(result);
        }
    }

    // Define module interface.

    {
        const auto result = impl->define();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return stopCreating(result);
        }
    }

    // Verify module taints.

    bool taintDiscontiguous = false;
    bool taintCrossDevice = false;

    if (impl->_taint != Module::Taint::CLEAN) {
        JST_TRACE("[MODULE] Module ('{}') is tainted. Verifying...", impl->_name);

        if ((impl->_taint & Taint::DISCONTIGUOUS) == Taint::DISCONTIGUOUS) {
            taintDiscontiguous = true;
        }

        if ((impl->_taint & Taint::CROSS_DEVICE) == Taint::CROSS_DEVICE) {
            taintCrossDevice = true;
        }
    }

    // Check if module provides all requested inputs.

    for (const auto& key : impl->_interface->inputs()) {
        if (!impl->_inputs.contains(key)) {
            JST_ERROR("[MODULE] Module '{}' requested missing input '{}'.", impl->_name, key);
            return stopCreating(Result::ERROR);
        }
    }

    // Verify input tensors device matches module device.

    for (const auto& [inputName, link] : inputs) {
        if (link.tensor.device() != impl->_device && !taintCrossDevice) {
            JST_ERROR("[MODULE] Input tensor device ('{}', DeviceType::{})"
                      " doesn't match the module device ('{}', DeviceType::{}).",
                      inputName, link.tensor.device(), impl->_name, impl->_device);
            return stopCreating(Result::ERROR);
        }

        if (!link.tensor.validShape()) {
            JST_ERROR("[MODULE] Input tensor ('{}') is invalid.", inputName);
            return stopCreating(Result::ERROR);
        }

        if (link.tensor.size() == 0) {
            JST_ERROR("[MODULE] Module ('{}') input tensor ('{}') size is zero.", impl->_name, inputName);
            return stopCreating(Result::ERROR);
        }

        if (!link.tensor.contiguous() && !taintDiscontiguous) {
            JST_ERROR("[MODULE] Contiguous tensor expected for module ('{}') input tensor ('{}').", impl->_name, inputName);
            return stopCreating(Result::ERROR);
        }
    }

    // Creating module.

    Result createResult;
#ifdef JST_OS_BROWSER
    if ((impl->_taint & Taint::BROWSER_MAIN_THREAD) == Taint::BROWSER_MAIN_THREAD) {
        std::pair<Impl*, Result*> ctx{impl.get(), &createResult};
        emscripten_proxy_sync(
            emscripten_proxy_get_system_queue(),
            emscripten_main_runtime_thread_id(),
            Impl::proxyCreate,
            &ctx);
    } else {
        createResult = impl->create();
    }
#else
    createResult = impl->create();
#endif

    // Check if module provides all requested outputs.

    if (createResult == Result::SUCCESS || createResult == Result::RELOAD) {
        for (const auto& key : impl->_interface->outputs()) {
            if (!impl->_outputs.contains(key)) {
                JST_ERROR("[MODULE] Module '{}' didn't create an expected output '{}'.", impl->_name, key);
                createResult = Result::ERROR;
                break;
            }
        }
    }

    if (createResult == Result::ERROR) {
        impl->_state = State::DESTROYING;
        const auto destroyResult = impl->destroyImplementation();
        if (destroyResult != Result::SUCCESS && destroyResult != Result::RELOAD) {
            impl->_state = State::ERRORED;
            JST_ERROR("[MODULE] Failed to clean up module '{}' after creation failure.", impl->_name);
        } else {
            impl->_state = State::DESTROYED;
        }
    }

    if (createResult != Result::SUCCESS && createResult != Result::RELOAD) {
        if (createResult != Result::ERROR) {
            impl->_state = (createResult == Result::INCOMPLETE)
                               ? State::INCOMPLETE
                               : State::ERRORED;
        }
        return createResult;
    }

    impl->_state = State::CREATED;
    return Result::SUCCESS;
}

Result Module::destroy() {
    if (impl->_state != State::CREATED &&
        impl->_state != State::INCOMPLETE &&
        impl->_state != State::ERRORED) {
        JST_ERROR("[MODULE] Cannot destroy module '{}' in its current lifecycle state.", impl->_name);
        return Result::ERROR;
    }

    impl->_state = State::DESTROYING;
    const auto result = impl->destroyImplementation();
    if (result != Result::SUCCESS && result != Result::RELOAD) {
        impl->_state = State::ERRORED;
        return result;
    }

    impl->_state = State::DESTROYED;
    return Result::SUCCESS;
}

Result Module::reconfigure(const Parser::Map& config, const bool& validateOnly) {
    if (impl->_state == State::DESTROYED || impl->_state == State::ERRORED) {
        return Result::RECREATE;
    }
    if (impl->_state != State::CREATED) {
        return Result::ERROR;
    }

    // Deserialize new configuration.

    {
        const auto result = impl->_candidateConfig->deserialize(config);
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return result;
        }
    }

    // Return early if the configuration is unchanged.

    if (impl->_candidateConfig->hash() == impl->_stagedConfig->hash()) {
        return Result::SUCCESS;
    }

    // Validate configuration and reconfigure the module if something changed.

    {
        const auto result = impl->validate();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return result;
        }
    }
    if (!validateOnly) {
        const auto result = impl->reconfigure();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            return result;
        }
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

Module::Timing Module::timing() const {
    return impl->_timing.get();
}

void Module::timing(const Timing& timing) {
    impl->_timing.publish(timing);
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

const Module::State& Module::state() const {
    return impl->_state;
}

const std::shared_ptr<Module::Surface>& Module::surface() {
    return impl->_surface;
}

}  // namespace Jetstream
