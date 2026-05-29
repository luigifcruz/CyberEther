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
                   std::unordered_set<std::string>& skippedModules) override;

 private:
    static inline std::shared_ptr<NativeCpuRuntimeContext> getRuntimeContext(const std::shared_ptr<Module>& module) {
        return std::dynamic_pointer_cast<NativeCpuRuntimeContext>(module->context()->runtime());
    }

    Runtime::Modules modulesMap;
    std::vector<std::string> moduleNames;
};

Result NativeCpuRuntime::create(const Runtime::Modules& modules) {
    // Initialize modules.

    modulesMap.clear();
    moduleNames.clear();

    for (const auto& [name, module] : modules) {
        if (module->device() != DeviceType::CPU || module->runtime() != RuntimeType::NATIVE) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Module '{}' is incompatible "
                      "(DeviceType::{}, RuntimeType::{}).", name, module->device(), module->runtime());
            return Result::ERROR;
        }

        JST_CHECK(getRuntimeContext(module)->computeInitialize());

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

Result NativeCpuRuntime::destroy() {
    for (auto& [_, module] : modulesMap) {
        JST_CHECK(getRuntimeContext(module)->computeDeinitialize());
    }

    modulesMap.clear();
    moduleNames.clear();

    return Result::SUCCESS;
}

Result NativeCpuRuntime::compute(const std::vector<std::string>& modules,
                                 std::unordered_set<std::string>& skippedModules) {
    const auto& targetNames = modules.empty() ? moduleNames : modules;

    for (const auto& name : targetNames) {
        if (!modulesMap.contains(name)) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Context for module '{}' not found.", name);
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
