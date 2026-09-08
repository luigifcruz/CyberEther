#ifndef JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

//
// Metadata [dependencies/metadata.cc]
//

struct PythonDependencyMetadata {
    std::vector<std::string> requirements;
    std::string requiresPython;

    bool operator==(const PythonDependencyMetadata&) const = default;
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

JETSTREAM_API Result ParsePythonDependencyPolicy(const std::string& value,
                                                 PythonDependencyPolicy& policy);
JETSTREAM_API PythonDependencyPolicy ConfiguredPythonDependencyPolicy();

//
// Environment [dependencies/environment.cc]
//

struct PythonDependencyEnvironment {
    std::string key;
    std::string sitePackagesPath;
};

JETSTREAM_API Result PreparePythonDependencyEnvironment(const std::vector<std::string>& requirements,
                                                        bool installIfMissing,
                                                        PythonDependencyEnvironment& environment,
                                                        std::function<void(std::string_view)> onOutput = {});

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH
