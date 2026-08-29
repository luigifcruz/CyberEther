#ifndef JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH

#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

struct PythonDependencyMetadata {
    std::vector<std::string> requirements;
    std::string requiresPython;
};

enum class PythonDependencyPolicy {
    Prompt,  // Ask for approval before installing dependencies.
    Allow,   // Install dependencies automatically.
    Deny,    // Never install dependencies.
};

struct PythonDependencyDecision {
    PythonDependencyPolicy policy = PythonDependencyPolicy::Prompt;
    bool installAllowed = false;
    bool consentRequired = false;
};

JETSTREAM_API Result ParsePythonDependencyMetadata(const std::string& source,
                                                   PythonDependencyMetadata& metadata);
JETSTREAM_API Result ValidatePythonDependencyMetadata(const PythonDependencyMetadata& metadata);

JETSTREAM_API Result ParsePythonDependencyPolicy(const std::string& value,
                                                 PythonDependencyPolicy& policy);
JETSTREAM_API PythonDependencyDecision ResolvePythonDependencyPolicy(const PythonDependencyMetadata& metadata,
                                                                     PythonDependencyPolicy policy);
JETSTREAM_API PythonDependencyDecision ResolvePythonDependencyPolicy(const PythonDependencyMetadata& metadata);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH
