#include "jetstream/parser.hh"
#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include <jetstream/block.hh>
#include <jetstream/detail/block_impl.hh>

namespace Jetstream {

Block::Block(const std::shared_ptr<Block::Impl>& impl,
             const std::shared_ptr<Block::Config>& stagedConfig,
             const std::shared_ptr<Block::Config>& candidateConfig) : impl(impl) {
    impl->_stagedConfig = stagedConfig;
    impl->_candidateConfig = candidateConfig;
}

Result Block::create(const std::string& name,
                     const DeviceType& device,
                     const RuntimeType& runtime,
                     const ProviderType& provider,
                     const Parser::Map& config,
                     const TensorMap& inputs,
                     const std::shared_ptr<Instance>& instance,
                     const std::shared_ptr<Render::Window>& render,
                     const std::shared_ptr<Scheduler>& scheduler) {
    JST_ASSERT(impl->_state == State::None || impl->_state == State::Destroyed,
               "[BLOCK] Cannot create block '{}' in state '{}'.", name, impl->_state);

    // Set implementation variables.

    impl->_state = State::Creating;

    impl->_inputs = inputs;
    impl->_outputs = TensorMap();
    impl->_interface = std::make_shared<Interface>();

    impl->_name = name;
    impl->_device = device;
    impl->_runtime = runtime;
    impl->_provider = provider;

    impl->_instance = instance;
    impl->_render = render;
    impl->_scheduler = scheduler;

    JST_DEBUG("[BLOCK] Creating block '{}'.", impl->_name);

    // Validate configuration.

    {
        const auto result = impl->_candidateConfig->deserialize(config);
        if (result != Result::SUCCESS) {
            impl->_state = State::Errored;
            return result;
        }
    }

    {
        const auto result = impl->validate();
        if (result != Result::SUCCESS && result != Result::RECREATE) {
            impl->_state = State::Errored;
            return result;
        }
    }

    // Commit candidate and configure block.

    {
        const auto result = impl->_stagedConfig->deserialize(config);
        if (result != Result::SUCCESS) {
            impl->_state = State::Errored;
            return result;
        }
    }

    {
        const auto result = impl->configure();
        if (result != Result::SUCCESS) {
            impl->_state = State::Errored;
            return result;
        }
    }

    // Define block interface.

    {
        const auto result = impl->define();
        if (result != Result::SUCCESS) {
            impl->_state = State::Errored;
            return result;
        }
    }

    // Check if block provides all requested inputs.

    for (const auto& [key, _] : impl->_interface->inputs()) {
        if (!impl->_inputs.contains(key)) {
            JST_ERROR("[BLOCK] Block '{}' has unconnected input '{}'.", impl->_name, key);
            impl->_diagnostic = JST_LOG_LAST_ERROR();
            impl->_state = State::Incomplete;
            return Result::INCOMPLETE;
        }

        const auto& link = impl->_inputs.at(key);
        if (!link.resolved()) {
            JST_ERROR("[BLOCK] Block '{}' has unresolved input '{}'.", impl->_name, key);
            impl->_diagnostic = JST_LOG_LAST_ERROR();
            impl->_state = State::Incomplete;
            return Result::INCOMPLETE;
        }
    }

    // Check if block received any undeclared inputs.

    for (const auto& [key, _] : impl->_inputs) {
        bool found = false;
        for (const auto& [inputKey, _] : impl->_interface->inputs()) {
            if (inputKey == key) {
                found = true;
                break;
            }
        }
        if (!found) {
            JST_ERROR("[BLOCK] Block '{}' received undeclared input '{}'.", impl->_name, key);
            impl->_diagnostic = JST_LOG_LAST_ERROR();
            impl->_state = State::Errored;
            return Result::ERROR;
        }
    }

    // Create block.

    const auto result = impl->create();

    if (result != Result::SUCCESS && result != Result::RELOAD) {
        impl->_diagnostic = JST_LOG_LAST_ERROR();
        impl->_state = (result == Result::INCOMPLETE) ? State::Incomplete : State::Errored;
        return result;
    }

    // Check if block provides all requested outputs.

    for (const auto& [key, _] : impl->_interface->outputs()) {
        if (!impl->_outputs.contains(key)) {
            JST_ERROR("[BLOCK] Block '{}' didn't create an expected output '{}'.", impl->_name, key);
            impl->_diagnostic = JST_LOG_LAST_ERROR();
            impl->_state = State::Errored;
            return Result::ERROR;
        }
    }

    impl->_state = State::Created;
    return Result::SUCCESS;
}

Result Block::destroy() {
    JST_DEBUG("[BLOCK] Destroying block '{}'.", impl->_name);
    JST_ASSERT(impl->_state == State::Created ||
               impl->_state == State::Incomplete ||
               impl->_state == State::Errored,
               "[BLOCK] Cannot destroy block '{}' in state '{}'.", impl->_name, impl->_state);

    impl->_state = State::Destroying;

    // Destroy internal modules in reverse.

    while (!impl->_moduleOrder.empty()) {
        const auto result = impl->moduleDestroy(impl->_moduleOrder.back());
        if (result != Result::SUCCESS) {
            impl->_state = State::Errored;
            return result;
        }
    }

    if (!impl->_modules.empty()) {
        JST_WARN("[BLOCK] Module '{}' still has {} child modules for destruction.",
                 impl->_name, impl->_modules.size());
    }

    impl->_state = State::Destroyed;

    return Result::SUCCESS;
}

Result Block::reconfigure(const Parser::Map& config) {
    // Deserialize the candidate configuration.

    JST_CHECK(impl->_candidateConfig->deserialize(config));

    // Return early if the configuration is unchanged.

    if (impl->_candidateConfig->hash() == impl->_stagedConfig->hash()) {
        JST_TRACE("[BLOCK] Configuration of '{}' is unchanged.", impl->_name);
        return Result::SUCCESS;
    }

    // Validate candidate configuration.

    JST_CHECK(impl->validate());

    // Backup previous configuration and commit.

    Parser::Map previousConfig;
    JST_CHECK(impl->_stagedConfig->serialize(previousConfig));
    JST_CHECK(impl->_stagedConfig->deserialize(config));

    // Run block configuration.

    JST_CHECK(impl->configure());

    // Validate all internal modules configurations.

    for (const auto& module : impl->_moduleOrder) {
        if (impl->moduleReconfigure(module, true) != Result::SUCCESS) {
            // Failed, reverting block to previous state.

            JST_CHECK(impl->_stagedConfig->deserialize(previousConfig));
            JST_CHECK(impl->configure());
            return Result::ERROR;
        }
    }

    // Reconfigure all internal modules.

    for (const auto& module : impl->_moduleOrder) {
        JST_CHECK(impl->moduleReconfigure(module));
    }

    // Refresh interface.

    impl->_interface = std::make_shared<Interface>();
    JST_CHECK(impl->define());

    return Result::SUCCESS;
}

Result Block::config(Parser::Map& config) const {
    return impl->_stagedConfig->serialize(config);
}

const Block::Config& Block::config() const {
    return *impl->_stagedConfig;
}

const TensorMap& Block::inputs() const {
    return impl->_inputs;
}

const TensorMap& Block::outputs() const {
    return impl->_outputs;
}

const std::string& Block::name() const {
    return impl->_name;
}

const DeviceType& Block::device() const {
    return impl->_device;
}

const RuntimeType& Block::runtime() const {
    return impl->_runtime;
}

const ProviderType& Block::provider() const {
    return impl->_provider;
}

const std::shared_ptr<Block::Interface>& Block::interface() const {
    return impl->_interface;
}

const Block::State& Block::state() const {
    return impl->_state;
}

const std::string& Block::diagnostic() const {
    return impl->_diagnostic;
}

const std::vector<std::shared_ptr<Module::Surface>>& Block::surfaces() const {
    return impl->surfaces();
}

const std::vector<std::string>& Block::modules() const {
    return impl->modules();
}

const char* GetBlockStateName(const Block::State& state) {
    switch (state) {
        case Block::State::None:
            return "none";
        case Block::State::Creating:
            return "creating";
        case Block::State::Created:
            return "created";
        case Block::State::Incomplete:
            return "incomplete";
        case Block::State::Errored:
            return "errored";
        case Block::State::Destroying:
            return "destroying";
        case Block::State::Destroyed:
            return "destroyed";
        default:
            return "unknown";
    }
}

const char* GetBlockStatePrettyName(const Block::State& state) {
    switch (state) {
        case Block::State::None:
            return "None";
        case Block::State::Creating:
            return "Creating";
        case Block::State::Created:
            return "Created";
        case Block::State::Incomplete:
            return "Incomplete";
        case Block::State::Errored:
            return "Errored";
        case Block::State::Destroying:
            return "Destroying";
        case Block::State::Destroyed:
            return "Destroyed";
        default:
            return "Unknown";
    }
}

}  // namespace Jetstream
