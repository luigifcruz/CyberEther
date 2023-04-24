#ifndef JETSTREAM_GRAPH_GENERIC_HH
#define JETSTREAM_GRAPH_GENERIC_HH

#include <memory>

#include "jetstream/memory/types.hh"
#include "jetstream/metadata.hh"
#include "jetstream/logger.hh"
#include "jetstream/module.hh"

namespace Jetstream { 

class Graph {
 public:
    virtual ~Graph() = default;

    const Result schedule(const std::shared_ptr<Compute>& block);
    virtual constexpr const Device device() const = 0;

    virtual const Result createCompute() = 0;
    virtual const Result compute() = 0;
    virtual const Result destroyCompute() = 0;

 protected:
    std::shared_ptr<RuntimeMetadata> metadata;
    std::vector<std::shared_ptr<Compute>> blocks;
};

}  // namespace Jetstream

#endif 
