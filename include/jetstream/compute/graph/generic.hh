#ifndef JETSTREAM_COMPUTE_GRAPH_GENERIC_HH
#define JETSTREAM_COMPUTE_GRAPH_GENERIC_HH

#include <set>
#include <memory>

#include "jetstream/memory/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/module.hh"

namespace Jetstream { 

class Graph {
 public:
    virtual ~Graph() = default;

    Result setModule(const std::shared_ptr<Compute>& block);

    Result setWiredInput(const U64& input);
    Result setWiredOutput(const U64& output);
    Result setExternallyWiredInput(const U64& input);
    Result setExternallyWiredOutput(const U64& output);

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
    virtual Result create() = 0;
    virtual Result compute() = 0;
    virtual Result computeReady() = 0;
    virtual Result destroy() = 0;

 protected:
    std::shared_ptr<Compute::Context> context;
    std::vector<std::shared_ptr<Compute>> blocks;
    std::set<U64> wiredInputSet;
    std::set<U64> wiredOutputSet;
    std::set<U64> externallyWiredInputSet;
    std::set<U64> externallyWiredOutputSet;
};

}  // namespace Jetstream

#endif 
