#ifndef JETSTREAM_MODULE_INTERFACE_HH
#define JETSTREAM_MODULE_INTERFACE_HH

#include "jetstream/module.hh"

namespace Jetstream {

struct Module::Interface {
 public:
    typedef std::vector<std::string> EntryList;

    Interface();
    ~Interface();

    const EntryList& inputs() const;
    const EntryList& outputs() const;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    friend struct Module::Impl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_MODULE_INTERFACE_HH
