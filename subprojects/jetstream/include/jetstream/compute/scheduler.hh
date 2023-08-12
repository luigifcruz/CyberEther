#ifndef JETSTREAM_COMPUTE_SCHEDULER_HH
#define JETSTREAM_COMPUTE_SCHEDULER_HH

#include <memory>
#include <stack>
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
              const std::unordered_map<std::string, BlockState>& blockStates,
              const std::vector<std::string>& blockStateOrder);
    ~Scheduler();

    Result create();
    Result compute();
    Result present();
    Result destroy();

    void drawDebugMessage() const;

 private:
    typedef std::vector<std::string> ExecutionOrder;
    typedef std::vector<std::pair<Device, ExecutionOrder>> DeviceExecutionOrder;

    struct ComputeModuleState {
        const BlockState* block;
        U64 clusterId;
        std::unordered_map<std::string, const Parser::VectorMetadata*> activeInputs;
        std::unordered_map<std::string, const Parser::VectorMetadata*> activeOutputs;
    };

    struct PresentModuleState {
        const BlockState* block;
    };

    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};

    std::shared_ptr<Render::Window>& window;
    std::vector<std::shared_ptr<Graph>> graphs;
    std::unordered_map<std::string, ComputeModuleState> computeModuleStates;
    std::unordered_map<std::string, PresentModuleState> presentModuleStates;
    DeviceExecutionOrder deviceExecutionOrder;

    Result removeInactive();
    Result arrangeDependencyOrder(ExecutionOrder& executionOrder);
    Result checkSequenceValidity(ExecutionOrder& executionOrder);
    Result createExecutionGraphs();
};

}  // namespace Jetstream

#endif
