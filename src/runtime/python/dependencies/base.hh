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

JETSTREAM_API Result ParsePythonDependencyMetadata(const std::string& source,
                                                   PythonDependencyMetadata& metadata);
JETSTREAM_API Result ValidatePythonDependencyMetadata(const PythonDependencyMetadata& metadata);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_DEPENDENCIES_BASE_HH
