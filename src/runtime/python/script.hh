#ifndef JETSTREAM_RUNTIME_PYTHON_SCRIPT_HH
#define JETSTREAM_RUNTIME_PYTHON_SCRIPT_HH

#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

struct PythonScriptMetadata {
    std::vector<std::string> dependencies;
    std::string requiresPython;
};

JETSTREAM_API Result ParsePythonScriptMetadata(const std::string& source,
                                               PythonScriptMetadata& metadata);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_SCRIPT_HH
