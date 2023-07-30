#ifndef JETSTREAM_STORE_HH
#define JETSTREAM_STORE_HH

#include "jetstream/parser.hh"
#include "jetstream/instance.hh"

namespace Jetstream {

typedef std::unordered_map<Parser::BackendIdentifier,
                           std::function<Result(Instance&, Parser::BackendRecord&)>,
                           Parser::BackendIdentifier::Hash, 
                           Parser::BackendIdentifier::Equal> BackendStore;

typedef std::unordered_map<Parser::ViewportIdentifier,
                           std::function<Result(Instance&, Parser::ViewportRecord&)>,
                           Parser::ViewportIdentifier::Hash, 
                           Parser::ViewportIdentifier::Equal> ViewportStore;

typedef std::unordered_map<Parser::ModuleIdentifier,
                           std::function<void(Instance&, Parser::ModuleRecord&)>,
                           Parser::ModuleIdentifier::Hash, 
                           Parser::ModuleIdentifier::Equal> ModuleStore;

typedef std::unordered_map<Parser::RenderIdentifier,
                           std::function<Result(Instance&, Parser::RenderRecord&)>,
                           Parser::RenderIdentifier::Hash, 
                           Parser::RenderIdentifier::Equal> RenderStore;

// MAYDO: Add new modules during runtime.
class JETSTREAM_API Store {
 public:
    Store(Store const&) = delete;
    void operator=(Store const&) = delete;

    static Store& GetInstance();

    static BackendStore& Backends() {
        return GetInstance().backends;
    }

    static ViewportStore& Viewports() {
        return GetInstance().viewports;
    }

    static RenderStore& Renders() {
        return GetInstance().renders;
    }

    static ModuleStore& Modules() {
        return GetInstance().modules;
    }

 private:
    Store();

    BackendStore backends;
    ViewportStore viewports;
    RenderStore renders;
    ModuleStore modules;

    BackendStore& defaultBackends();
    ViewportStore& defaultViewports();
    RenderStore& defaultRenders();
    ModuleStore& defaultModules();
};

}  // namespace Jetstream

#endif