#include <jetstream/detail/module_interface_impl.hh>

namespace Jetstream {

Module::Interface::Interface() {
    impl = std::make_shared<Impl>();
}

Module::Interface::~Interface() {
    impl.reset();
}

const Module::Interface::EntryList& Module::Interface::inputs() const {
    return impl->inputs;
}

const Module::Interface::EntryList& Module::Interface::outputs() const {
    return impl->outputs;
}

}  // namespace Jetstream
