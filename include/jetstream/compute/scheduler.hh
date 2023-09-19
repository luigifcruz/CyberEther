#ifndef JETSTREAM_COMPUTE_SCHEDULER_HH
#define JETSTREAM_COMPUTE_SCHEDULER_HH

#include <memory>
#include <stack>
#include <ranges>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "jetstream/state.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

class JETSTREAM_API Scheduler {
 public:
    Result addModule(const Locale& locale, const std::shared_ptr<BlockState>& block);
    Result removeModule(const Locale& locale);
    Result compute();
    Result present();
    Result destroy();

    void drawDebugMessage() const;

 private:
    typedef std::vector<std::string> ExecutionOrder;
    typedef std::vector<std::pair<Device, ExecutionOrder>> DeviceExecutionOrder;

    struct ComputeModuleState {
        std::shared_ptr<BlockState> block;
        U64 clusterId;
        std::unordered_map<std::string, const Parser::Record*> activeInputs;
        std::unordered_map<std::string, const Parser::Record*> activeOutputs;
    };

    struct PresentModuleState {
        std::shared_ptr<BlockState> block;
    };

    std::atomic_flag computeSync = ATOMIC_FLAG_INIT;
    std::atomic_flag presentSync = ATOMIC_FLAG_INIT;
    std::atomic_flag computeWait = ATOMIC_FLAG_INIT;
    std::atomic_flag computeHalt = ATOMIC_FLAG_INIT;
    std::atomic_flag presentHalt = ATOMIC_FLAG_INIT;

    std::unordered_map<std::string, ComputeModuleState> computeModuleStates;
    std::unordered_map<std::string, PresentModuleState> presentModuleStates;

    std::unordered_map<std::string, ComputeModuleState> validComputeModuleStates;
    std::unordered_map<std::string, PresentModuleState> validPresentModuleStates;

    std::vector<std::shared_ptr<Graph>> graphs;
    ExecutionOrder executionOrder;
    DeviceExecutionOrder deviceExecutionOrder;

    Result removeInactive();
    Result arrangeDependencyOrder();
    Result checkSequenceValidity();
    Result createExecutionGraphs();

    void lock();
    void unlock();
};

}  // namespace Jetstream

#endif
