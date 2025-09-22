#ifndef JETSTREAM_MODULE_IMPL_HH
#define JETSTREAM_MODULE_IMPL_HH

#include "jetstream/module.hh"
#include "jetstream/module_context.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/runtime.hh"

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

struct Module::Impl {
 public:
    virtual ~Impl() = default;

#ifdef JST_OS_BROWSER
    static void proxyCreate(void* arg);
    static void proxyDestroy(void* arg);
#endif

 protected:
    // Lifecycle

    virtual Result validate();
    virtual Result define();
    virtual Result create();
    virtual Result destroy();
    virtual Result reconfigure();

    // Identity

    const std::string& name() const;
    const DeviceType& device() const;
    const RuntimeType& runtime() const;
    const ProviderType& provider() const;
    const Module::Taint& taint() const;

    Result defineTaint(const Taint& taint);

    // I/O

    const TensorMap& inputs() const;
    TensorMap& outputs();

    Result defineInterfaceInput(const std::string& key);
    Result defineInterfaceOutput(const std::string& key);

    // Components

    std::shared_ptr<Render::Window>& render();

    // Surface

    Result surfaceCreateManifest(SurfaceManifest&& manifest);
    Result surfaceUpdateManifestSize(const std::string& id, const Extent2D<U64>& size);
    std::vector<MouseEvent> surfaceConsumeMouseEvents();
    std::vector<SurfaceEvent> surfaceConsumeSurfaceEvents();

 private:
    // Identity

    std::string _name;
    DeviceType _device;
    RuntimeType _runtime;
    ProviderType _provider;
    Taint _taint;

    // I/O

    TensorMap _inputs;
    TensorMap _outputs;
    std::shared_ptr<Interface> _interface;

    // Components

    std::shared_ptr<Render::Window> _render;
    std::shared_ptr<Module::Context> _context;
    std::shared_ptr<Module::Surface> _surface;

    // Configuration

    std::shared_ptr<Module::Config> _stagedConfig;
    std::shared_ptr<Module::Config> _candidateConfig;

    friend class Module;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MODULE_IMPL_HH
