#ifndef JETSTREAM_BLOCK_INTERFACE_HH
#define JETSTREAM_BLOCK_INTERFACE_HH

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/block.hh"
#include "jetstream/parser.hh"

namespace Jetstream {

struct JETSTREAM_API Block::Interface {
 public:
    struct Entry {
        std::string label;
        Parser::Map format;
        std::string help;
        std::function<std::any()> metric;
    };

    typedef std::vector<std::pair<std::string, Entry>> EntryList;

    static Result ValidateFormat(const Parser::Map& format, bool metric = false);

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
