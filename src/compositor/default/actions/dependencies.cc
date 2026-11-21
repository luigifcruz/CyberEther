#include "dependencies.hh"

namespace Jetstream {

DependencyActions::DependencyActions(DefaultCompositorCallbacks& callbacks) :
    callbacks(callbacks) {}

Result DependencyActions::handle(const MailInstallPythonDependencies& msg) {
    callbacks.enqueueCommand([generation = msg.generation]() -> Result {
        return BeginPythonDependencyInstallation(generation);
    }, true);
    return Result::SUCCESS;
}

}  // namespace Jetstream
