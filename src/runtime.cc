#include "jetstream/runtime.hh"
#include "jetstream/detail/runtime_impl.hh"

namespace Jetstream {

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
std::shared_ptr<Runtime::Impl> NativeCpuRuntimeFactory();
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

const std::shared_ptr<Runtime::Metrics>& Runtime::metrics() const {
    return impl->metrics();
}

Result Runtime::compute(const std::vector<std::string>& modules) {
    if (impl->presentRunning.load(std::memory_order_acquire)) {
        JST_ERROR("[RUNTIME] Cannot call compute() while present() is running.");
        return Result::ERROR;
    }

    impl->computeRunning.store(true, std::memory_order_release);
    Result result = impl->compute(modules);
    impl->computeRunning.store(false, std::memory_order_release);

    return result;
}

const char* GetRuntimeName(const RuntimeType& runtime) {
    switch (runtime) {
        case RuntimeType::NATIVE:
            return "native";
        case RuntimeType::MLIR:
            return "mlir";
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
        default:
            return "None";
    }
}

RuntimeType StringToRuntime(const std::string& runtime) {
    if (runtime == "native") {
        return RuntimeType::NATIVE;
    } else if (runtime == "mlir") {
        return RuntimeType::MLIR;
    } else {
        return RuntimeType::NONE;
    }
}

}  // namespace Jetstream
