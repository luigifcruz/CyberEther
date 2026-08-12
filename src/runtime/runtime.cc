#include "jetstream/runtime.hh"
#include "jetstream/detail/runtime_impl.hh"
#include "jetstream/module.hh"
#include "jetstream/runtime_context.hh"

namespace Jetstream {

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
std::shared_ptr<Runtime::Impl> NativeCpuRuntimeFactory();
std::shared_ptr<Runtime::Impl> PythonRuntimeFactory();
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
std::shared_ptr<Runtime::Impl> NativeCudaRuntimeFactory();
#endif

Runtime::Runtime(const std::string& name, const DeviceType& device, const RuntimeType& type) {
    switch (device) {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        case DeviceType::CPU:
            switch (type) {
                case RuntimeType::NATIVE:
                    impl = NativeCpuRuntimeFactory();
                    impl->name = name;
                    impl->device = device;
                    impl->backend = type;
                    return;
                case RuntimeType::PYTHON:
                    impl = PythonRuntimeFactory();
                    impl->name = name;
                    impl->device = device;
                    impl->backend = type;
                    return;
                case RuntimeType::MLIR:
                    return;
                default:
                    JST_FATAL("[RUNTIME] Unknown runtime type.");
                    throw Result::FATAL;
            }
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        case DeviceType::CUDA:
            switch (type) {
                case RuntimeType::NATIVE:
                    impl = NativeCudaRuntimeFactory();
                    impl->name = name;
                    impl->device = device;
                    impl->backend = type;
                    return;
                case RuntimeType::MLIR:
                    return;
                default:
                    JST_FATAL("[RUNTIME] Unknown runtime type.");
                    throw Result::FATAL;
            }
#endif
        default:
            JST_FATAL("[RUNTIME] Unknown device type.");
            throw Result::FATAL;
    }
}

Result Runtime::create(const Modules& modules) {
    JST_DEBUG("[RUNTIME] Creating runtime.");
    return impl->create(modules);
}

Result Runtime::destroy() {
    JST_DEBUG("[RUNTIME] Destroying runtime.");
    return impl->destroy();
}

Runtime::Context::Diagnostic Runtime::Context::diagnostic() const {
    return {};
}

Result Runtime::compute(const std::vector<std::string>& modules,
                        std::unordered_set<std::string>& skippedModules,
                        std::unordered_set<std::string>& failedModules) {
    failedModules.clear();

    if (impl->presentRunning.load(std::memory_order_acquire)) {
        JST_ERROR("[RUNTIME] Cannot call compute() while present() is running.");
        return Result::ERROR;
    }

    impl->computeRunning.store(true, std::memory_order_release);
    Result result = impl->compute(modules, skippedModules, failedModules);
    impl->computeRunning.store(false, std::memory_order_release);

    if (result == Result::SUCCESS) {
        failedModules.clear();
    }

    return result;
}

bool Runtime::Impl::hasSkippedInputs(const std::shared_ptr<Module>& module,
                                     const std::unordered_set<std::string>& skippedModules) {
    for (const auto& [_, link] : module->inputs()) {
        if (!link.producer.has_value()) {
            continue;
        }

        if (skippedModules.contains(link.producer->module)) {
            return true;
        }
    }

    return false;
}

const char* GetRuntimeName(const RuntimeType& runtime) {
    switch (runtime) {
        case RuntimeType::NATIVE:
            return "native";
        case RuntimeType::MLIR:
            return "mlir";
        case RuntimeType::PYTHON:
            return "python";
        default:
            return "none";
    }
}

const char* GetRuntimePrettyName(const RuntimeType& runtime){
    switch (runtime) {
        case RuntimeType::NATIVE:
            return "Native";
        case RuntimeType::MLIR:
            return "MLIR";
        case RuntimeType::PYTHON:
            return "Python";
        default:
            return "None";
    }
}

RuntimeType StringToRuntime(const std::string& runtime) {
    if (runtime == "native") {
        return RuntimeType::NATIVE;
    } else if (runtime == "mlir") {
        return RuntimeType::MLIR;
    } else if (runtime == "python") {
        return RuntimeType::PYTHON;
    } else {
        return RuntimeType::NONE;
    }
}

}  // namespace Jetstream
