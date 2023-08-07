#ifndef JETSTREAM_STATE_HH
#define JETSTREAM_STATE_HH

#include <string>
#include <functional>
#include <unordered_set>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/interface.hh"
#include "jetstream/parser.hh"

namespace Jetstream {

struct RenderState {
    Parser::RenderRecord record;
};

struct ViewportState {
    Parser::ViewportRecord record;
};

struct BackendState {
    Parser::BackendRecord record;
};

struct BlockState {
    Parser::ModuleRecord record;
    std::function<Parser::RecordMap()> getConfigFunc;
    std::function<Parser::RecordMap()> getInputFunc;
    std::function<Parser::RecordMap()> getOutputFunc;

    std::shared_ptr<Module> module;
    std::shared_ptr<Compute> compute;
    std::shared_ptr<Present> present;

    std::shared_ptr<Interface> interface;
};

}  // namespace Jetstream

#endif