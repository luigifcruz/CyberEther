#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_DEPENDENCIES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_DEPENDENCIES_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"

#include "runtime/python/dependencies/coordinator.hh"

#include <tuple>

namespace Jetstream {

struct DependencyActions {
    using Filter = std::tuple<MailInstallPythonDependencies>;

    DefaultCompositorCallbacks& callbacks;

    explicit DependencyActions(DefaultCompositorCallbacks& callbacks);
    Result handle(const MailInstallPythonDependencies& msg);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_DEPENDENCIES_HH
