#ifndef JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH
#define JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH

#include <string>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

struct PythonRuntimeContext;
struct PythonDependencyEnvironment;

struct PythonDependencySnapshot {
    std::vector<std::string> requirements;
    U64 generation = 0;
};

JETSTREAM_API Result StagePythonDependencies(PythonRuntimeContext* context,
                                             const std::vector<std::string>& requirements);
JETSTREAM_API Result SchedulePythonDependencies(PythonRuntimeContext* context);
JETSTREAM_API Result UnschedulePythonDependencies(PythonRuntimeContext* context);
JETSTREAM_API Result RemovePythonDependencies(PythonRuntimeContext* context);
JETSTREAM_API PythonDependencySnapshot SnapshotPythonDependencies();
JETSTREAM_API Result ActivatePythonDependencyEnvironment(const PythonDependencyEnvironment& environment);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH
