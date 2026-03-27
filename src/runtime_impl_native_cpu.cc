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

    Result compute(const std::vector<std::string>& modules = {}) override;

    const std::shared_ptr<Runtime::Metrics>& metrics() const final;

 private:
    static inline std::shared_ptr<NativeCpuRuntimeContext> getRuntimeContext(const std::shared_ptr<Module>& module) {
        return std::dynamic_pointer_cast<NativeCpuRuntimeContext>(module->context()->runtime());
    }

    Runtime::Modules modulesMap;
    std::shared_ptr<Runtime::Metrics> runtimeMetrics;
};

Result NativeCpuRuntime::create(const Runtime::Modules& modules) {
    // Setup metrics.

    runtimeMetrics = std::make_shared<Runtime::Metrics>();
    runtimeMetrics->runtime = name;
    runtimeMetrics->device = GetDevicePrettyName(device);
    runtimeMetrics->backend = GetRuntimePrettyName(backend);

    // Initialize modules.

    for (const auto& [name, module] : modules) {
        if (module->device() != DeviceType::CPU || module->runtime() != RuntimeType::NATIVE) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Module '{}' is incompatible "
                      "(DeviceType::{}, RuntimeType::{}).", name, module->device(), module->runtime());
            return Result::ERROR;
        }

        JST_CHECK(getRuntimeContext(module)->computeInitialize());
        modulesMap[name] = module;
    }

    return Result::SUCCESS;
}

Result NativeCpuRuntime::destroy() {
    for (auto& [_, module] : modulesMap) {
        JST_CHECK(getRuntimeContext(module)->computeDeinitialize());
    }

    modulesMap.clear();
    runtimeMetrics.reset();

    return Result::SUCCESS;
}

const std::shared_ptr<Runtime::Metrics>& NativeCpuRuntime::metrics() const {
    return runtimeMetrics;
}

Result NativeCpuRuntime::compute(const std::vector<std::string>& modules) {
    // Build target list.

    std::unordered_map<std::string, std::shared_ptr<Module>> targets;

    if (modules.empty()) {
        targets = modulesMap;
    } else {
        for (const auto& name : modules) {
            if (!modulesMap.contains(name)) {
                JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Context for module '{}' not found.", name);
                return Result::ERROR;
            }
            targets[name] = modulesMap.at(name);
        }
    }

    // Execute modules.

    for (auto& [name, module] : targets) {
        const auto start = std::chrono::steady_clock::now();

        JST_CHECK(getRuntimeContext(module)->computeSubmit());

        const auto end = std::chrono::steady_clock::now();

        auto& cycles = runtimeMetrics->cycles[name];
        auto& averageComputeTime = runtimeMetrics->averageComputeTime[name];

        const F32 elapsedMs = std::chrono::duration<F32, std::milli>(end - start).count();
        const F32 totalTime = averageComputeTime * static_cast<F32>(cycles++);
        averageComputeTime = (totalTime + elapsedMs) / static_cast<F32>(cycles);
    }

    return Result::SUCCESS;
}

std::shared_ptr<Runtime::Impl> NativeCpuRuntimeFactory() {
    return std::make_shared<NativeCpuRuntime>();
}

}  // namespace Jetstream
