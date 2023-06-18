#ifndef JETSTREAM_GRAPH_GENERIC_HH
#define JETSTREAM_GRAPH_GENERIC_HH

#include <set>
#include <memory>

#include "jetstream/memory/types.hh"
#include "jetstream/metadata.hh"
#include "jetstream/logger.hh"
#include "jetstream/module.hh"

namespace Jetstream { 

class Graph {
 public:
    virtual ~Graph() = default;

    Result setModule(const std::shared_ptr<Compute>& block);

    Result setWiredInputs(const std::vector<U64>& inputs);
    Result setWiredOutputs(const std::vector<U64>& outputs);
    Result setExternallyWiredInputs(const std::vector<U64>& inputs);
    Result setExternallyWiredOutputs(const std::vector<U64>& outputs);

    constexpr const std::set<U64>& getWiredInputs() const {
        return wiredInputSet;
    }

    constexpr const std::set<U64>& getWiredOutputs() const {
        return wiredOutputSet;
    }

    constexpr const std::set<U64>& getExternallyWiredInputs() const {
        return externallyWiredInputSet;
    }

    constexpr const std::set<U64>& getExternallyWiredOutputs() const {
        return externallyWiredOutputSet;
    }

    virtual constexpr Device device() const = 0;
    virtual Result createCompute() = 0;
    virtual Result compute() = 0;
    virtual Result destroyCompute() = 0;

 protected:
    std::shared_ptr<RuntimeMetadata> metadata;
    std::vector<std::shared_ptr<Compute>> blocks;
    std::set<U64> wiredInputSet;
    std::set<U64> wiredOutputSet;
    std::set<U64> externallyWiredInputSet;
    std::set<U64> externallyWiredOutputSet;
};

}  // namespace Jetstream

#endif 
