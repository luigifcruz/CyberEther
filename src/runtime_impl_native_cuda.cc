#include <jetstream/detail/runtime_impl.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/module.hh>
#include <chrono>

namespace Jetstream {

struct NativeCudaRuntime : public Runtime::Impl {
 public:
    virtual ~NativeCudaRuntime() = default;

    Result create(const Runtime::Modules& modules) override;
    Result destroy() override;

    Result compute(const std::vector<std::string>& modules,
                   std::unordered_set<std::string>& skippedModules) override;

    const std::shared_ptr<Runtime::Metrics>& metrics() const final;

 private:
    static inline std::shared_ptr<NativeCudaRuntimeContext> getRuntimeContext(const std::shared_ptr<Module>& module) {
        return std::dynamic_pointer_cast<NativeCudaRuntimeContext>(module->context()->runtime());
    }

    Runtime::Modules modulesMap;
    std::vector<std::string> moduleNames;
    std::shared_ptr<Runtime::Metrics> runtimeMetrics;

    cudaStream_t stream;
    std::unordered_map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> eventMap;
};

Result NativeCudaRuntime::create(const Runtime::Modules& modules) {
    // Setup metrics.

    runtimeMetrics = std::make_shared<Runtime::Metrics>();
    runtimeMetrics->runtime = name;
    runtimeMetrics->device = GetDevicePrettyName(device);
    runtimeMetrics->backend = GetRuntimePrettyName(backend);

    // Create stream.

    JST_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), [&]{
        JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't create stream: {}", err);
    });

    // Initialize modules.

    modulesMap.clear();
    moduleNames.clear();

    for (const auto& [name, module] : modules) {
        // Check module and runtime compatibility.

        if (module->device() != DeviceType::CUDA || module->runtime() != RuntimeType::NATIVE) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Module '{}' is incompatible "
                      "(DeviceType::{}, RuntimeType::{}).", name, module->device(), module->runtime());
            return Result::ERROR;
        }

        // Initialize module.

        JST_CHECK(getRuntimeContext(module)->computeInitialize());

        // Create events for module.

        cudaEvent_t startEvent;
        cudaEvent_t endEvent;

        JST_CUDA_CHECK(cudaEventCreate(&startEvent), [&]{
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't create start event for '{}': {}", name, err);
        });

        JST_CUDA_CHECK(cudaEventCreate(&endEvent), [&]{
            cudaEventDestroy(startEvent);
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't create end event for '{}': {}", name, err);
        });

        // Save module state.

        modulesMap[name] = module;
        moduleNames.push_back(name);
        eventMap[name] = {startEvent, endEvent};
    }

    return Result::SUCCESS;
}

Result NativeCudaRuntime::destroy() {
    // Deinitalize modules.

    for (auto& [name, module] : modulesMap) {
        // Deinitialize module.

        JST_CHECK(getRuntimeContext(module)->computeDeinitialize());

        // Destroy events.

        auto& [startEvent, endEvent] = eventMap[name];

        JST_CUDA_CHECK(cudaEventDestroy(startEvent), [&]{
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't destroy start event for '{}': {}", name, err);
        });

        JST_CUDA_CHECK(cudaEventDestroy(endEvent), [&]{
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't destroy end event for '{}': {}", name, err);
        });
    }

    // Destroy stream.

    JST_CUDA_CHECK(cudaStreamDestroy(stream), [&]{
        JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't destroy stream: {}", err);
    });

    // Clear module state.

    modulesMap.clear();
    moduleNames.clear();
    eventMap.clear();
    runtimeMetrics.reset();

    return Result::SUCCESS;
}

const std::shared_ptr<Runtime::Metrics>& NativeCudaRuntime::metrics() const {
    return runtimeMetrics;
}

Result NativeCudaRuntime::compute(const std::vector<std::string>& modules,
                                  std::unordered_set<std::string>& skippedModules) {
    const auto& targetNames = modules.empty() ? moduleNames : modules;
    std::vector<std::string> executedModules;

    for (const auto& name : targetNames) {
        if (!modulesMap.contains(name)) {
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Context for module '{}' not found.", name);
            return Result::ERROR;
        }

        const auto& module = modulesMap.at(name);

        if (hasSkippedInputs(module, skippedModules)) {
            skippedModules.insert(name);
            continue;
        }

        auto& [startEvent, endEvent] = eventMap[name];

        JST_CUDA_CHECK(cudaEventRecord(startEvent, stream), [&]{
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't record start event for '{}': {}", name, err);
        });

        const auto result = getRuntimeContext(module)->computeSubmit(stream);

        if (result == Result::YIELD || result == Result::TIMEOUT) {
            return result;
        }

        if (result != Result::SUCCESS && result != Result::RELOAD && result != Result::SKIP) {
            return result;
        }

        JST_CUDA_CHECK(cudaEventRecord(endEvent, stream), [&]{
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't record end event for '{}': {}", name, err);
        });

        JST_CUDA_CHECK(cudaGetLastError(), [&]{
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Module kernel execution failed: {}", err);
        });

        executedModules.push_back(name);

        if (result == Result::SKIP) {
            skippedModules.insert(name);
        }
    }

    // Wait for all blocks to finish.

    JST_CUDA_CHECK(cudaStreamSynchronize(stream), [&]{
        JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't synchronize stream: {}", err);
    });

    // Check for any errors.

    JST_CUDA_CHECK(cudaGetLastError(), [&]{
        JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Runtime execution failed: {}", err);
    });

    // Register timings.

    for (const auto& name : executedModules) {
        auto& [startEvent, endEvent] = eventMap[name];

        F32 elapsedMs = 0;
        JST_CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent), [&]{
            JST_ERROR("[RUNTIME_IMPL_NATIVE_CUDA] Can't get elapsed time for '{}': {}", name, err);
        });

        auto& cycles = runtimeMetrics->cycles[name];
        auto& averageComputeTime = runtimeMetrics->averageComputeTime[name];

        const F32 totalTime = averageComputeTime * static_cast<F32>(cycles++);
        averageComputeTime = (totalTime + elapsedMs) / static_cast<F32>(cycles);
    }

    return Result::SUCCESS;
}

std::shared_ptr<Runtime::Impl> NativeCudaRuntimeFactory() {
    return std::make_shared<NativeCudaRuntime>();
}

}  // namespace Jetstream
