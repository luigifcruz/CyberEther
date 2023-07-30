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
};

}  // namespace Jetstream

#endif
