#ifndef JETSTREAM_STATE_HH
#define JETSTREAM_STATE_HH

#include <string>
#include <functional>
#include <unordered_set>
#include <unordered_map>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/bundle.hh"
#include "jetstream/module.hh"
#include "jetstream/interface.hh"
#include "jetstream/parser.hh"

namespace Jetstream {

struct BlockState {
    bool complete;

    Parser::ModuleRecord record;

    std::shared_ptr<Bundle> bundle;
    std::shared_ptr<Module> module;

    std::shared_ptr<Compute> compute;
    std::shared_ptr<Present> present;
    std::shared_ptr<Interface> interface;
};

}  // namespace Jetstream

#endif
