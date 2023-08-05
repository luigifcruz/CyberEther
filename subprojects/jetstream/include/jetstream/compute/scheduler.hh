#ifndef JETSTREAM_COMPUTE_SCHEDULER_HH
#define JETSTREAM_COMPUTE_SCHEDULER_HH

#include <memory>
#include <ranges>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "jetstream/state.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

class JETSTREAM_API Scheduler {
 public:
    Scheduler(std::shared_ptr<Render::Window>& window,
              std::unordered_map<std::string, BlockState>& blockStates,
              std::unordered_map<U64, std::string>& blockStateMap);
    ~Scheduler();

    Result create();
    Result compute();
    Result present();
    Result destroy();

    constexpr const U64& getNumberOfGraphs() const {
        return _graphCount;
    }

    constexpr const U64& getNumberOfComputeBlocks() const {
        return _computeBlockCount;
    }

    constexpr const U64& getNumberOfPresentBlocks() const {
        return _presentBlockCount;
    }

    constexpr const std::unordered_set<Device>& getComputeDevices() const {
        return _computeDevices;
    }

 private:
    std::unordered_map<std::string, BlockState>& blockStates;
    std::unordered_map<U64, std::string>& blockStateMap;
    std::shared_ptr<Render::Window>& window;

    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};

    std::vector<std::string> executionOrder;
    std::vector<std::pair<Device, std::vector<std::string>>> deviceExecutionOrder;

    std::vector<std::shared_ptr<Graph>> graphs;

    Result printGraphSummary();
    Result filterStaleIo();
    Result applyTopologicalSort();
    Result createComputeGraphs();
    Result assertInplaceCorrectness();

    U64 _computeBlockCount;
    U64 _presentBlockCount;
    U64 _graphCount;
    std::unordered_set<Device> _computeDevices;
};

}  // namespace Jetstream

#endif
