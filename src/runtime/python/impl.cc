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
                   std::unordered_set<std::string>& stalledModules,
                   std::unordered_set<std::string>& failedModules) override;

 private:
    static inline std::shared_ptr<PythonRuntimeContext> getRuntimeContext(const std::shared_ptr<Module>& module) {
        return std::dynamic_pointer_cast<PythonRuntimeContext>(module->context()->runtime());
    }

    struct Entry {
        std::shared_ptr<Module> module;
        std::shared_ptr<PythonRuntimeContext> context;
    };

    std::unordered_map<std::string, Entry> modulesMap;
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

        const auto initializeResult = context->computeInitialize();
        if (initializeResult != Result::SUCCESS &&
            initializeResult != Result::RELOAD) {
            const auto cleanup = [&](const std::string& moduleName,
                                     const std::shared_ptr<PythonRuntimeContext>& moduleContext) {
                const auto result = moduleContext->computeDeinitialize();
                if (result != Result::SUCCESS && result != Result::RELOAD) {
                    JST_ERROR("[RUNTIME_IMPL_PYTHON] Failed to roll back module '{}' "
                              "while creating runtime '{}'.", moduleName, this->name);
                }
            };

            cleanup(name, context);
            for (auto it = moduleNames.rbegin(); it != moduleNames.rend(); ++it) {
                cleanup(*it, modulesMap.at(*it).context);
            }
            modulesMap.clear();
            moduleNames.clear();
            return initializeResult;
        }

        Module::Timing timing;
        timing.runtime = this->name;
        timing.device = GetDevicePrettyName(device);
        timing.backend = GetRuntimePrettyName(backend);
        module->timing(timing);

        modulesMap[name] = {module, context};
        moduleNames.push_back(name);
    }

    return Result::SUCCESS;
}

Result PythonRuntime::destroy() {
    Result result = Result::SUCCESS;
    for (auto it = moduleNames.rbegin(); it != moduleNames.rend(); ++it) {
        const auto deinitializeResult =
            modulesMap.at(*it).context->computeDeinitialize();
        if (result == Result::SUCCESS && deinitializeResult != Result::SUCCESS &&
            deinitializeResult != Result::RELOAD) {
            result = deinitializeResult;
        }
    }

    modulesMap.clear();
    moduleNames.clear();

    return result;
}

Result PythonRuntime::compute(const std::vector<std::string>& modules,
                              std::unordered_set<std::string>& skippedModules,
                              std::unordered_set<std::string>& stalledModules,
                              std::unordered_set<std::string>& failedModules) {
    const auto& targetNames = modules.empty() ? moduleNames : modules;

    for (const auto& name : targetNames) {
        const auto entry = modulesMap.find(name);
        if (entry == modulesMap.end()) {
            failedModules.insert(name);
            JST_ERROR("[RUNTIME_IMPL_PYTHON] Context for module '{}' not found.", name);
            return Result::ERROR;
        }

        const auto& module = entry->second.module;

        if (skippedModules.contains(name) || stalledModules.contains(name)) {
            continue;
        }

        if (hasSkippedInputs(module, skippedModules, stalledModules)) {
            stalledModules.insert(name);
            continue;
        }

        const auto start = std::chrono::steady_clock::now();
        const auto result = entry->second.context->computeSubmit();
        const auto end = std::chrono::steady_clock::now();

        if (result == Result::YIELD || result == Result::TIMEOUT) {
            stalledModules.insert(name);
            continue;
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
