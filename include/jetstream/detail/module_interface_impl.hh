#ifndef JETSTREAM_DETAIL_MODULE_INTERFACE_IMPL_HH
#define JETSTREAM_DETAIL_MODULE_INTERFACE_IMPL_HH

#include "../module_interface.hh"

namespace Jetstream {

struct Module::Interface::Impl {
    Module::Interface::EntryList inputs;
    Module::Interface::EntryList outputs;
};

}  // namespace Jetstream

#endif  // JETSTREAM_DETAIL_MODULE_INTERFACE_IMPL_HH
