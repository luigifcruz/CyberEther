#ifndef JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH
#define JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH

#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

struct PythonRuntimeContext;

struct PythonDependencySnapshot {
    std::vector<std::string> requirements;
    U64 generation = 0;
};

JETSTREAM_API Result StagePythonDependencies(const PythonRuntimeContext* context,
                                             const std::vector<std::string>& requirements);
JETSTREAM_API Result SchedulePythonDependencies(const PythonRuntimeContext* context);
JETSTREAM_API Result UnschedulePythonDependencies(const PythonRuntimeContext* context);
JETSTREAM_API Result RemovePythonDependencies(const PythonRuntimeContext* context);
JETSTREAM_API PythonDependencySnapshot SnapshotPythonDependencies();

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH
