#ifndef JETSTREAM_BLOCK_INTERFACE_HH
#define JETSTREAM_BLOCK_INTERFACE_HH

#include <any>
#include <functional>
#include <memory>

#include "jetstream/block.hh"

namespace Jetstream {

struct Block::Interface {
 public:
    struct Entry {
        std::string label;
        std::string format;
        std::string help;
        std::function<std::any()> metric;
    };

    typedef std::vector<std::pair<std::string, Entry>> EntryList;

    Interface();
    ~Interface();

    const EntryList& configs() const;
    const EntryList& inputs() const;
    const EntryList& outputs() const;
    const EntryList& metrics() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend struct Block::Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_BLOCK_INTERFACE_HH
