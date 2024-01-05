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

#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

class JETSTREAM_API Scheduler {
 public:
    Result addModule(const Locale& locale, 
                     const std::shared_ptr<Module>& module,
                     const Parser::RecordMap& inputMap,
                     const Parser::RecordMap& outputMap,
                     std::shared_ptr<Compute>& compute,
                     std::shared_ptr<Present>& present);
    Result removeModule(const Locale& locale);
    Result compute();
    Result present();
    Result destroy();

    void drawDebugMessage() const;

 private:
    typedef std::vector<std::string> ExecutionOrder;
    typedef std::vector<std::pair<Device, ExecutionOrder>> DeviceExecutionOrder;

    struct ComputeModuleState {
        std::shared_ptr<Compute> module;
        Parser::RecordMap inputMap;
        Parser::RecordMap outputMap;
        Device device;
        U64 clusterId;
        std::unordered_map<std::string, const Parser::Record*> activeInputs;
        std::unordered_map<std::string, const Parser::Record*> activeOutputs;
    };

    struct PresentModuleState {
        std::shared_ptr<Present> module;
        Parser::RecordMap inputMap;
        Parser::RecordMap outputMap;
    };

    std::mutex sharedMutex;
    std::condition_variable presentCond;
    std::condition_variable computeCond;
    bool computeSync = false;
    bool presentSync = false;

    std::atomic_flag computeWait{false};
    std::atomic_flag computeHalt{true};
    std::atomic_flag presentHalt{true};

    std::unordered_map<std::string, ComputeModuleState> computeModuleStates;
    std::unordered_map<std::string, PresentModuleState> presentModuleStates;

    std::unordered_map<std::string, ComputeModuleState> validComputeModuleStates;
    std::unordered_map<std::string, PresentModuleState> validPresentModuleStates;

    bool running = true;
    std::vector<std::shared_ptr<Graph>> graphs;
    ExecutionOrder executionOrder;
    DeviceExecutionOrder deviceExecutionOrder;

    Result removeInactive();
    Result arrangeDependencyOrder();
    Result checkSequenceValidity();
    Result createExecutionGraphs();

    Result lockState(const std::function<Result()>& func);
};

}  // namespace Jetstream

#endif
