#ifndef JETSTREAM_DOMAINS_CORE_PYTHON_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_PYTHON_MODULE_IMPL_HH

#include <jetstream/detail/module_impl.hh>
#include <jetstream/domains/core/python/module.hh>

namespace Jetstream::Modules {

struct PythonImpl : public Module::Impl, public DynamicConfig<Python> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    static std::string inputPortName(U64 index);
    static std::string outputPortName(U64 index);
    static void normalizeOutputSpecs(Python& config);
    Module::Interface::EntryList inputPortOrder() const;
    Module::Interface::EntryList outputPortOrder() const;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_PYTHON_MODULE_IMPL_HH
