#include <jetstream/detail/block_interface_impl.hh>

namespace Jetstream {

Block::Interface::Interface() {
    impl = std::make_shared<Impl>();
}

Block::Interface::~Interface() {
    impl.reset();
}

const Block::Interface::EntryList& Block::Interface::configs() const {
    return impl->configs;
}

const Block::Interface::EntryList& Block::Interface::inputs() const {
    return impl->inputs;
}

const Block::Interface::EntryList& Block::Interface::outputs() const {
    return impl->outputs;
}

const Block::Interface::EntryList& Block::Interface::metrics() const {
    return impl->metrics;
}

}  // namespace Jetstream
