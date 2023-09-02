#ifndef JETSTREAM_STORE_HH
#define JETSTREAM_STORE_HH

#include "jetstream/parser.hh"
#include "jetstream/instance.hh"

namespace Jetstream {

typedef std::unordered_map<Parser::BackendFingerprint,
                           std::function<Result(Instance&, Parser::BackendRecord&)>,
                           Parser::BackendFingerprint::Hash,
                           Parser::BackendFingerprint::Equal> BackendStore;

typedef std::unordered_map<Parser::ViewportFingerprint,
                           std::function<Result(Instance&, Parser::ViewportRecord&)>,
                           Parser::ViewportFingerprint::Hash,
                           Parser::ViewportFingerprint::Equal> ViewportStore;

typedef std::unordered_map<Parser::ModuleFingerprint,
                           std::function<Result(Instance&, Parser::ModuleRecord&)>,
                           Parser::ModuleFingerprint::Hash,
                           Parser::ModuleFingerprint::Equal> ModuleStore;

typedef std::unordered_map<Parser::RenderFingerprint,
                           std::function<Result(Instance&, Parser::RenderRecord&)>,
                           Parser::RenderFingerprint::Hash,
                           Parser::RenderFingerprint::Equal> RenderStore;

struct ModuleListValueStore {
    bool isBundle;
    std::string title;
    std::string small;
    std::string detailed;
    std::map<Device, std::vector<std::tuple<std::string, std::string, std::string>>> options;
};

typedef std::unordered_map<std::string, ModuleListValueStore> ModuleListStore;

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

    static ModuleListStore& ModuleList(const std::string& filter = "") {
        return GetInstance()._moduleList(filter);
    }

 private:
    Store();

    BackendStore backends;
    ViewportStore viewports;
    RenderStore renders;

    ModuleStore modules;
    ModuleListStore moduleList;
    ModuleListStore filteredModuleList;

    BackendStore& defaultBackends();
    ViewportStore& defaultViewports();
    RenderStore& defaultRenders();
    ModuleStore& defaultModules();
    ModuleListStore& defaultModuleList();

    ModuleListStore& _moduleList(const std::string& filter);
};

}  // namespace Jetstream

#endif
