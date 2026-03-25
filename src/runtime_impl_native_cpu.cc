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
    Runtime::Modules modulesMap;
    std::shared_ptr<Runtime::Metrics> runtimeMetrics;
};

Result NativeCpuRuntime::create(const Runtime::Modules& modules) {
    runtimeMetrics = std::make_shared<Runtime::Metrics>();
    runtimeMetrics->runtime = name;
    runtimeMetrics->device = GetDevicePrettyName(device);
    runtimeMetrics->backend = GetRuntimePrettyName(backend);
    F32 totalInitTime = 0.0f;

    for (const auto& [name, module] : modules) {
        if (module->device() != DeviceType::CPU || module->runtime() != RuntimeType::NATIVE) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Module '{}' is incompatible (DeviceType::{}, Runtime::{}).", name, module->device(), module->runtime());
            return Result::ERROR;
        }

        const auto& ctx = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(module->context()->runtime());

        const auto start = std::chrono::steady_clock::now();
        JST_CHECK(ctx->computeInitialize());
        const auto end = std::chrono::steady_clock::now();

        modulesMap[name] = module;
        totalInitTime += std::chrono::duration<F32, std::milli>(end - start).count();
    }

    runtimeMetrics->initializationTime = totalInitTime;

    return Result::SUCCESS;
}

Result NativeCpuRuntime::destroy() {
    for (auto& [_, module] : modulesMap) {
        const auto& ctx = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(module->context()->runtime());
        JST_CHECK(ctx->computeDeinitialize());
    }

    modulesMap.clear();
    runtimeMetrics.reset();

    return Result::SUCCESS;
}

const std::shared_ptr<Runtime::Metrics>& NativeCpuRuntime::metrics() const {
    return runtimeMetrics;
}

Result NativeCpuRuntime::compute(const std::vector<std::string>& modules) {
    const auto start = std::chrono::steady_clock::now();

    if (modules.empty()) {
        // Compute all if module list is empty.

        for (auto& [name, module] : modulesMap) {
            const auto& ctx = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(module->context()->runtime());
            JST_CHECK(ctx->computeSubmit());
        }
    } else {
        // Compute specific modules if provided.

        for (const auto& name : modules) {
            if (!modulesMap.contains(name)) {
                JST_ERROR("[RUNTIME_IMPL_NATIVE_CPU] Context for module '{}' not found.", name);
                return Result::ERROR;
            }
            const auto& ctx = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(modulesMap.at(name)->context()->runtime());
            JST_CHECK(ctx->computeSubmit());
        }
    }

    const auto end = std::chrono::steady_clock::now();

    const F32 elapsedMs = std::chrono::duration<F32, std::milli>(end - start).count();
    const F32 totalTime = runtimeMetrics->averageComputeTime * static_cast<F32>(runtimeMetrics->cycles);
    runtimeMetrics->cycles += 1;
    runtimeMetrics->averageComputeTime = (totalTime + elapsedMs) / static_cast<F32>(runtimeMetrics->cycles);

    return Result::SUCCESS;
}

std::shared_ptr<Runtime::Impl> NativeCpuRuntimeFactory() {
    return std::make_shared<NativeCpuRuntime>();
}

}  // namespace Jetstream
