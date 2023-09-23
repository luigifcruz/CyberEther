#ifndef JETSTREAM_STORE_HH
#define JETSTREAM_STORE_HH

#include "jetstream/parser.hh"
#include "jetstream/instance.hh"

namespace Jetstream {

typedef std::unordered_map<Parser::ModuleFingerprint,
                           std::function<Result(Instance&, Parser::ModuleRecord&)>,
                           Parser::ModuleFingerprint::Hash,
                           Parser::ModuleFingerprint::Equal> ModuleStore;

struct ModuleListValueStore {
    bool isBundle;
    std::string title;
    std::string small;
    std::string detailed;
    std::map<Device, std::vector<std::tuple<std::string, std::string, std::string>>> options;
};

typedef std::unordered_map<std::string, ModuleListValueStore> ModuleListStore;

struct FlowgraphListValueStore {
    std::string title;
    std::string description;
    const char* data;
};

typedef std::unordered_map<std::string, FlowgraphListValueStore> FlowgraphListStore;

// MAYDO: Add new modules during runtime.
class JETSTREAM_API Store {
 public:
    Store(Store const&) = delete;
    void operator=(Store const&) = delete;

    static Store& GetInstance();

    static ModuleStore& Modules() {
        return GetInstance().modules;
    }

    static ModuleListStore& ModuleList(const std::string& filter = "", const bool& showModules = true) {
        GetInstance()._moduleList(filter, showModules);
        return GetInstance().filteredModuleList;
    }

    static FlowgraphListStore& FlowgraphList(const std::string& filter = "") {
        GetInstance()._flowgraphList(filter);
        return GetInstance().filteredFlowgraphList;
    }

 private:
    Store();

    ModuleStore modules;
    ModuleListStore moduleList;
    ModuleListStore filteredModuleList;
    std::string lastModuleListFilter;
    bool lastModuleListShowModules;

    FlowgraphListStore flowgraphList;
    FlowgraphListStore filteredFlowgraphList;
    std::string lastFlowgraphListFilter;
    
    const ModuleStore& defaultModules();
    const ModuleListStore& defaultModuleList();
    const FlowgraphListStore& defaultFlowgraphList();

    Result _moduleList(const std::string& filter, const bool& modules);
    Result _flowgraphList(const std::string& filter);
};

}  // namespace Jetstream

#endif
