#include <jetstream/detail/runtime_impl.hh>
#include <jetstream/module.hh>
#include <jetstream/module_context.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/runtime_context_python.hh>
#include <jetstream/scheduler_context.hh>

#include <chrono>

namespace Jetstream {

struct PythonRuntime : public Runtime::Impl {
 public:
    virtual ~PythonRuntime() = default;

    Result create(const Runtime::Modules& modules) override;
    Result destroy() override;

    Result compute(const std::vector<std::string>& modules,
                   std::unordered_set<std::string>& skippedModules,
                   std::unordered_set<std::string>& failedModules) override;

 private:
    static inline std::shared_ptr<PythonRuntimeContext> getRuntimeContext(const std::shared_ptr<Module>& module) {
        return std::dynamic_pointer_cast<PythonRuntimeContext>(module->context()->runtime());
    }

    Runtime::Modules modulesMap;
    std::vector<std::string> moduleNames;
};

Result PythonRuntime::create(const Runtime::Modules& modules) {
    modulesMap.clear();
    moduleNames.clear();

    for (const auto& [name, module] : modules) {
        if (module->device() != DeviceType::CPU || module->runtime() != RuntimeType::PYTHON) {
            JST_ERROR("[RUNTIME_IMPL_PYTHON] Module '{}' is incompatible "
                      "(DeviceType::{}, RuntimeType::{}).", name, module->device(), module->runtime());
            return Result::ERROR;
        }

        const auto context = getRuntimeContext(module);
        if (!context) {
            JST_ERROR("[RUNTIME_IMPL_PYTHON] Module '{}' does not provide a Python runtime context.", name);
            return Result::ERROR;
        }

        JST_CHECK(context->computeInitialize());

        Module::Timing timing;
        timing.runtime = this->name;
        timing.device = GetDevicePrettyName(device);
        timing.backend = GetRuntimePrettyName(backend);
        module->timing(timing);

        modulesMap[name] = module;
        moduleNames.push_back(name);
    }

    return Result::SUCCESS;
}

Result PythonRuntime::destroy() {
    for (auto& [_, module] : modulesMap) {
        JST_CHECK(getRuntimeContext(module)->computeDeinitialize());
    }

    modulesMap.clear();
    moduleNames.clear();

    return Result::SUCCESS;
}

Result PythonRuntime::compute(const std::vector<std::string>& modules,
                              std::unordered_set<std::string>& skippedModules,
                              std::unordered_set<std::string>& failedModules) {
    const auto& targetNames = modules.empty() ? moduleNames : modules;

    for (const auto& name : targetNames) {
        if (!modulesMap.contains(name)) {
            failedModules.insert(name);
            JST_ERROR("[RUNTIME_IMPL_PYTHON] Context for module '{}' not found.", name);
            return Result::ERROR;
        }

        const auto& module = modulesMap.at(name);

        if (hasSkippedInputs(module, skippedModules)) {
            skippedModules.insert(name);
            continue;
        }

        const auto start = std::chrono::steady_clock::now();
        const auto result = getRuntimeContext(module)->computeSubmit();
        const auto end = std::chrono::steady_clock::now();

        if (result == Result::YIELD || result == Result::TIMEOUT) {
            return result;
        }

        if (result != Result::SUCCESS && result != Result::RELOAD && result != Result::SKIP) {
            failedModules.insert(name);
            return result;
        }

        const F32 elapsedMs = std::chrono::duration<F32, std::milli>(end - start).count();

        auto timing = module->timing();
        timing.cycles += 1;
        timing.computeTime += elapsedMs;
        module->timing(timing);

        if (result == Result::SKIP) {
            skippedModules.insert(name);
        }
    }

    return Result::SUCCESS;
}

std::shared_ptr<Runtime::Impl> PythonRuntimeFactory() {
    return std::make_shared<PythonRuntime>();
}

}  // namespace Jetstream
