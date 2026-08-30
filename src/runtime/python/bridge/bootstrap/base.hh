#ifndef JETSTREAM_RUNTIME_PYTHON_BRIDGE_BOOTSTRAP_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_BRIDGE_BOOTSTRAP_BASE_HH

#include <string>

#include "jetstream/types.hh"

namespace Jetstream {

Result SwitchPythonEnvironmentPath(const std::string& previousPath,
                                   const std::string& currentPath);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_BRIDGE_BOOTSTRAP_BASE_HH
