#ifndef JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH

#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

//
// Metadata [dependencies/metadata.cc]
//

struct PythonDependencyMetadata {
    std::vector<std::string> requirements;
    std::string requiresPython;
};

JETSTREAM_API Result ParsePythonDependencyMetadata(const std::string& source,
                                                   PythonDependencyMetadata& metadata);

//
// Preflight [dependencies/preflight.cc]
//

JETSTREAM_API Result ValidatePythonDependencyMetadata(const PythonDependencyMetadata& metadata);

//
// Policy [dependencies/policy.cc]
//

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

JETSTREAM_API Result ParsePythonDependencyPolicy(const std::string& value,
                                                 PythonDependencyPolicy& policy);
JETSTREAM_API PythonDependencyDecision ResolvePythonDependencyPolicy(const PythonDependencyMetadata& metadata,
                                                                     PythonDependencyPolicy policy);
JETSTREAM_API PythonDependencyDecision ResolvePythonDependencyPolicy(const PythonDependencyMetadata& metadata);

//
// Environment [dependencies/environment.cc]
//

struct PythonDependencyEnvironment {
    std::vector<std::string> requirements;
    std::string key;
    std::string sitePackagesPath;
};

JETSTREAM_API Result PreparePythonDependencyEnvironment(const std::vector<std::string>& requirements,
                                                        bool installIfMissing,
                                                        PythonDependencyEnvironment& environment);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH
