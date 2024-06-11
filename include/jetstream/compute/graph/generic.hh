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

    Result setModule(const std::shared_ptr<Compute>& block, 
                     const std::unordered_set<U64>& inputSet,
                     const std::unordered_set<U64>& outputSet);

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
    virtual Result compute(std::unordered_set<U64>& yielded) = 0;
    virtual Result computeReady() = 0;
    virtual Result destroy() = 0;

    static void Yield(std::unordered_set<U64>& yielded, const std::unordered_set<U64>& outputSet);
    static bool ShouldYield(std::unordered_set<U64>& yielded, const std::unordered_set<U64>& inputSet);

 protected:
    struct ComputeUnit {
        std::shared_ptr<Compute> block;
        std::unordered_set<U64> inputSet;
        std::unordered_set<U64> outputSet;
    };

    std::shared_ptr<Compute::Context> context;
    std::vector<ComputeUnit> computeUnits;
    std::set<U64> wiredInputSet;
    std::set<U64> wiredOutputSet;
    std::set<U64> externallyWiredInputSet;
    std::set<U64> externallyWiredOutputSet;
};

}  // namespace Jetstream

#endif 
