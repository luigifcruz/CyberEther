#ifndef JETSTREAM_DETAIL_BLOCK_INTERFACE_IMPL_HH
#define JETSTREAM_DETAIL_BLOCK_INTERFACE_IMPL_HH

#include "../block_interface.hh"

namespace Jetstream {

struct Block::Interface::Impl {
    Block::Interface::EntryList configs;
    Block::Interface::EntryList inputs;
    Block::Interface::EntryList outputs;
    Block::Interface::EntryList metrics;
};

}  // namespace Jetstream

#endif  // JETSTREAM_DETAIL_BLOCK_INTERFACE_IMPL_HH
