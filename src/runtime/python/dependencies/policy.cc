#include "runtime/python/dependencies/base.hh"

#include <string>

#include "jetstream/logger.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/backend/base.hh"
#endif

namespace Jetstream {

namespace {

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
std::string ConfiguredDependencyPolicy() {
    return Backend::State<DeviceType::CPU>()->getDependencyPolicy();
}
#else
std::string ConfiguredDependencyPolicy() {
    return {};
}
#endif

}  // namespace

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

PythonDependencyDecision ResolvePythonDependencyPolicy(const PythonDependencyMetadata& metadata,
                                                       PythonDependencyPolicy policy) {
    if (metadata.requirements.empty()) {
        return {.policy = policy};
    }

    const auto declared = jst::fmt::format("{} requirement(s)",
                                           metadata.requirements.size());

    switch (policy) {
        case PythonDependencyPolicy::Allow:
            JST_INFO("[RUNTIME_CONTEXT_PYTHON] The script declares PEP 723 metadata with {}; "
                     "dependency installation is allowed by policy.",
                     declared);
            return {.policy = policy, .installAllowed = true};
        case PythonDependencyPolicy::Prompt:
            JST_INFO("[RUNTIME_CONTEXT_PYTHON] The script declares PEP 723 metadata with {}; "
                     "dependency installation is allowed after approval by policy.",
                     declared);
            return {.policy = policy, .installAllowed = true, .consentRequired = true};
        case PythonDependencyPolicy::Deny:
            JST_WARN("[RUNTIME_CONTEXT_PYTHON] The script declares PEP 723 metadata with {}; "
                     "dependency installation is disabled by policy.",
                     declared);
            return {.policy = policy};
    }

    return {.policy = policy};
}

PythonDependencyDecision ResolvePythonDependencyPolicy(const PythonDependencyMetadata& metadata) {
    const auto configured = ConfiguredDependencyPolicy();

    PythonDependencyPolicy policy = PythonDependencyPolicy::Prompt;
    if (!configured.empty()) {
        if (ParsePythonDependencyPolicy(configured, policy) != Result::SUCCESS) {
            JST_WARN("[RUNTIME_CONTEXT_PYTHON] Ignoring the invalid configured dependency "
                     "policy '{}'. Falling back to 'prompt'.",
                     configured);
            policy = PythonDependencyPolicy::Prompt;
        }
    }

    return ResolvePythonDependencyPolicy(metadata, policy);
}

}  // namespace Jetstream
