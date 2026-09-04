#include "runtime/python/dependencies/base.hh"

#include <string>

#include "jetstream/logger.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/backend/base.hh"
#endif

namespace Jetstream {

Result ParsePythonDependencyPolicy(const std::string& value, PythonDependencyPolicy& policy) {
    if (value == "prompt") {
        policy = PythonDependencyPolicy::Prompt;
        return Result::SUCCESS;
    }
    if (value == "allow") {
        policy = PythonDependencyPolicy::Allow;
        return Result::SUCCESS;
    }
    if (value == "deny") {
        policy = PythonDependencyPolicy::Deny;
        return Result::SUCCESS;
    }

    JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Invalid dependency policy '{}'. "
              "Expected one of: prompt, allow, deny.",
              value);
    return Result::ERROR;
}

PythonDependencyPolicy ConfiguredPythonDependencyPolicy() {
    PythonDependencyPolicy policy = PythonDependencyPolicy::Prompt;
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    const auto& configured = Backend::State<DeviceType::CPU>()->getDependencyPolicy();
    if (!configured.empty() &&
        ParsePythonDependencyPolicy(configured, policy) != Result::SUCCESS) {
        JST_WARN("[RUNTIME_CONTEXT_PYTHON] Ignoring the invalid configured dependency "
                 "policy '{}'. Falling back to 'prompt'.", configured);
    }
#endif
    return policy;
}

}  // namespace Jetstream
