#ifndef JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH
#define JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH

#include <memory>
#include <string>
#include <vector>

#include "jetstream/flowgraph.hh"
#include "jetstream/types.hh"

namespace Jetstream {

struct PythonRuntimeContext;
struct PythonDependencyEnvironment;

struct PythonDependencySnapshot {
    std::vector<std::string> requirements;
    U64 generation = 0;
};

enum class PythonDependencyRequestState {
    None,
    ApprovalRequired,
    Denied,
    Installing,
    Installed,
    Failed,
};

struct PythonDependencyRequestEntry {
    std::string requirement;
    std::string block;
    std::weak_ptr<Flowgraph::View> view;
};

struct PythonDependencyRequest {
    U64 generation = 0;
    PythonDependencyRequestState state = PythonDependencyRequestState::None;
    std::vector<PythonDependencyRequestEntry> dependencies;
    std::string output;
    std::string message;
    bool approved = false;
};

struct PythonDependencyStateSnapshot {
    U64 generation = 0;
    PythonDependencyRequest request;
};

JETSTREAM_API Result StagePythonDependencies(PythonRuntimeContext* context,
                                             const std::vector<std::string>& requirements);
JETSTREAM_API Result SetPythonDependencyPolicy(const std::string& value);
JETSTREAM_API Result SetPythonDependencyOrigin(PythonRuntimeContext* context,
                                               const std::string& block,
                                               const std::shared_ptr<Flowgraph::View>& view);
JETSTREAM_API Result SchedulePythonDependencies(PythonRuntimeContext* context);
JETSTREAM_API Result UnschedulePythonDependencies(PythonRuntimeContext* context);
JETSTREAM_API Result RemovePythonDependencies(PythonRuntimeContext* context);
JETSTREAM_API PythonDependencySnapshot SnapshotPythonDependencies();
JETSTREAM_API PythonDependencyStateSnapshot SnapshotPythonDependencyState();
JETSTREAM_API PythonDependencyRequest GetPythonDependencyRequest();
JETSTREAM_API Result ReconcilePythonDependencies();
JETSTREAM_API Result BeginPythonDependencyInstallation(U64 generation);
JETSTREAM_API Result InstallPythonDependencies(U64 generation);
JETSTREAM_API Result ActivatePythonDependencyEnvironment(const PythonDependencyEnvironment& environment);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_CONTEXT_HH
