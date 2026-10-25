#include <jetstream/detail/runtime_impl.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/module.hh>
#include <chrono>

namespace Jetstream {

struct NativeCpuRuntime : public Runtime::Impl {
 public:
    virtual ~NativeCpuRuntime() = default;

    Result create(const Runtime::Modules& modules) override;
    Result destroy() override;

    Result compute(const std::vector<std::string>& modules,
                   std::unordered_set<std::string>& skippedModules,
                   std::unordered_set<std::string>& stalledModules,
                   std::unordered_set<std::string>& failedModules) override;

 private:
    static inline std::shared_ptr<NativeCpuRuntimeContext> getRuntimeContext(const std::shared_ptr<Module>& module) {
        return std::dynamic_pointer_cast<NativeCpuRuntimeContext>(module->context()->runtime());
    }

    struct Entry {
        std::shared_ptr<Module> module;
        std::shared_ptr<NativeCpuRuntimeContext> context;
    };

    std::unordered_map<std::string, Entry> modulesMap;
    std::vector<std::string> moduleNames;
};

Result NativeCpuRuntime::create(const Runtime::Modules& modules) {
    // Initialize modules.

    modulesMap.clear();
    moduleNames.clear();

    const auto cleanupModule = [&](const std::string& name,
                                   const std::shared_ptr<Module>& module) -> void {
        const auto result = getRuntimeContext(module)->computeDeinitialize();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Failed to clean up module '{}' while creating runtime '{}'.",
                      name, this->name);
        }
    };

    const auto cleanupRuntime = [&]() -> void {
        const auto result = destroy();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Failed to clean up partially created runtime '{}'.",
                      this->name);
        }
    };

    for (const auto& [name, module] : modules) {
        if (module->device() != DeviceType::CPU || module->runtime() != RuntimeType::NATIVE) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Module '{}' is incompatible "
                      "(DeviceType::{}, RuntimeType::{}).", name, module->device(), module->runtime());
            cleanupRuntime();
            return Result::ERROR;
        }

        const auto context = getRuntimeContext(module);
        const auto result = context->computeInitialize();
        if (result != Result::SUCCESS && result != Result::RELOAD) {
            cleanupModule(name, module);
            cleanupRuntime();
            return result;
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

Result NativeCpuRuntime::destroy() {
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

Result NativeCpuRuntime::compute(const std::vector<std::string>& modules,
                                 std::unordered_set<std::string>& skippedModules,
                                 std::unordered_set<std::string>& stalledModules,
                                 std::unordered_set<std::string>& failedModules) {
    const auto& targetNames = modules.empty() ? moduleNames : modules;

    for (const auto& name : targetNames) {
        const auto entry = modulesMap.find(name);
        if (entry == modulesMap.end()) {
            failedModules.insert(name);
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Context for module '{}' not found.", name);
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

std::shared_ptr<Runtime::Impl> NativeCpuRuntimeFactory() {
    return std::make_shared<NativeCpuRuntime>();
}

}  // namespace Jetstream
