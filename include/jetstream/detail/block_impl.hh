#ifndef JETSTREAM_BLOCK_IMPL_HH
#define JETSTREAM_BLOCK_IMPL_HH

#include <any>
#include <functional>

#include "jetstream/block.hh"
#include "jetstream/block_interface.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/registry.hh"

namespace Jetstream {

#ifndef JETSTREAM_DYNAMIC_CONFIG_DEFINED
#define JETSTREAM_DYNAMIC_CONFIG_DEFINED
template<typename ConfigType>
class DynamicConfig : public ConfigType {
 public:
    DynamicConfig() : ConfigType() {
        _candidate = std::make_shared<ConfigType>();
    }

    std::shared_ptr<ConfigType>& candidate() {
        return _candidate;
    }

 private:
    std::shared_ptr<ConfigType> _candidate;
};
#endif  // JETSTREAM_DYNAMIC_CONFIG_DEFINED

struct Block::Impl {
 public:
    virtual ~Impl() = default;

 protected:
    // Lifecycle

    virtual Result validate();
    virtual Result configure();
    virtual Result define();
    virtual Result create();
    virtual Result destroy();

    // Identity

    const std::string& name() const;
    const DeviceType& device() const;
    const RuntimeType& runtime() const;
    const ProviderType& provider() const;

    // I/O

    const TensorMap& inputs() const;
    TensorMap& outputs();

    Result defineInterfaceInput(const std::string& key,
                                const std::string& label,
                                const std::string& help);
    Result defineInterfaceOutput(const std::string& key,
                                 const std::string& label,
                                 const std::string& help);
    Result defineInterfaceConfig(const std::string& key,
                                 const std::string& label,
                                 const std::string& help,
                                 const std::string& format);
    Result defineInterfaceMetric(const std::string& key,
                                 const std::string& label,
                                 const std::string& help,
                                 const std::string& format,
                                 std::function<std::any()> metric);

    // Components

    std::shared_ptr<Instance>& instance();
    std::shared_ptr<Render::Window>& render();
    std::shared_ptr<Scheduler>& scheduler();
    const std::vector<std::shared_ptr<Module::Surface>>& surfaces() const;

    // Modules

    Result moduleCreate(const std::string name,
                        const std::shared_ptr<Module::Config>& config,
                        const TensorMap& inputs);
    Result moduleDestroy(const std::string name);
    Result moduleExposeOutput(const std::string blockPort,
                              const std::pair<std::string, std::string>& moduleOutput);
    TensorLink moduleGetOutput(const std::pair<std::string, std::string>& moduleOutput);
    Result moduleReconfigure(const std::string name, const bool& validateOnly = false);
    std::shared_ptr<Module> moduleHandle(const std::string& name);

 private:
    // Identity

    std::string _name;
    DeviceType _device;
    RuntimeType _runtime;
    ProviderType _provider;
    Block::State _state;
    std::string _diagnostic;

    // I/O

    TensorMap _inputs;
    TensorMap _outputs;
    std::shared_ptr<Interface> _interface;

    // Components

    std::shared_ptr<Instance> _instance;
    std::shared_ptr<Render::Window> _render;
    std::shared_ptr<Scheduler> _scheduler;
    std::vector<std::shared_ptr<Module::Surface>> _surfaces;

    // Modules

    struct ModuleEntry {
        std::shared_ptr<Module> module;
        std::shared_ptr<Module::Config> config;
    };
    std::unordered_map<std::string, ModuleEntry> _modules;
    std::vector<std::string> _moduleOrder;

    // Configuration

    std::shared_ptr<Block::Config> _stagedConfig;
    std::shared_ptr<Block::Config> _candidateConfig;

    friend class Block;
};

}  // namespace Jetstream

#endif  // JETSTREAM_BLOCK_IMPL_HH
