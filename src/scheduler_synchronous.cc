// Synchronous Scheduler Implementation
//
// Coordinates compute and present threads with the following guarantees:
//   - Present waits at most kPresentWait for compute to reach a segment boundary, then
//     skips the frame. It always withdraws its request before returning.
//   - Present has priority: compute parks at segment boundaries while present wants in,
//     and never waits for a later frame. Pause releases a park before the first segment,
//     a cycle that already started keeps yielding until it completes.
//   - Present capture copies compute data under the data lock, submit runs every
//     frame without it and is only excluded by state modifications.
//   - State modifications (add/remove/reload) safely halt both threads.
//   - Compute polling (hasPendingCompute) doesn't hold locks, allowing fast state changes.
//   - Branches run independently: a module runs only when every input is fresh, and a
//     producer waits until every consumer has read its output.
//   - Every output carries the source frames it derives from, and a join only pairs inputs
//     whose shared sources agree. Outputs only move forward, so an input older than what
//     another input has already delivered can never pair and is discarded, emitting an
//     empty output for the discarded frames.
//   - A skip emits an empty output that flows downstream without compute. A module that
//     receives one waits for a frame on every input and drops them all together, so every
//     module consumes exactly one frame per input per step. A throttled module that is not
//     due drops its frames the same way.
//   - Idle cycles report their next throttle deadline instead of sleeping, so the instance
//     sleeps once per round only when every flowgraph is idle.

#include <jetstream/detail/scheduler_impl.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/runtime.hh>
#include <jetstream/module.hh>
#include <jetstream/instance.hh>
#include <jetstream/viewport/adapters/generic.hh>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <ranges>
#include <shared_mutex>
#include <thread>

namespace Jetstream {

namespace {

constexpr auto kPresentWait = std::chrono::milliseconds(2);
constexpr auto kPresentPoll = std::chrono::microseconds(250);

}  // namespace

struct SynchronousScheduler : public Scheduler::Impl {
 public:
    Result create() override;
    Result destroy() override;
    Result start() override;
    Result stop() override;
    Result pause() override;
    Result resume() override;
    bool paused() const override;

    Result add(const std::shared_ptr<Module>& module) override;
    Result remove(const std::shared_ptr<Module>& module) override;
    Result reload(const std::shared_ptr<Module>& module) override;
    Result synchronize(const std::function<Result()>& fn) override;

    Result present(std::unordered_set<std::string>& failedModules) override;
    Result compute(std::unordered_set<std::string>& failedModules) override;

 private:
    Result capture(std::unordered_set<std::string>& failedModules);
    Result submit(std::unordered_set<std::string>& failedModules);
    void haltAll();
    void resumeAll();
    bool isComputeHalted() const;
    bool isPresentHalted() const;
    void waitComputeHalt();
    U64 currentGeneration() const;
    void incrementGeneration();
    void requestPresentEntry();
    void clearPresentRequest();
    bool isPresentRequested() const;
    bool isStopping() const;
    void setComputeActive();
    void clearComputeActive();
    bool isComputeActive() const;
    void setPresentActive();
    void clearPresentActive();
    std::shared_lock<std::shared_mutex> sharedDataLock();
    std::shared_lock<std::shared_mutex> trySharedDataLock();
    std::unique_lock<std::mutex> exclusiveSharedLock();
    std::unique_lock<std::mutex> tryExclusiveSharedLock();
    void waitForPresentToFinish(std::unique_lock<std::mutex>& lock);
    bool waitForPresentTurn(std::chrono::steady_clock::time_point deadline);
    void yieldToPresent(bool started);
    void notifyCompute();
    void notifyPresent();
    void wakePresent();
    Result lockState(const std::function<Result()>& func);

    std::atomic_flag computeHalt = ATOMIC_FLAG_INIT;
    std::atomic_flag presentHalt = ATOMIC_FLAG_INIT;
    std::shared_mutex dataMutex;
    std::mutex sharedMutex;
    std::mutex surfaceMutex;
    std::condition_variable presentCond;
    std::condition_variable computeCond;
    bool presentSync = false;
    bool computeSync = false;
    std::atomic<bool> presentWantsIn{false};
    std::atomic<bool> presentable{false};
    std::mutex handoffMutex;
    std::condition_variable handoffCond;
    std::atomic<bool> stopping{false};
    std::atomic<bool> pausing{false};
    std::atomic<U64> generation{0};
    std::weak_ptr<Viewport::Generic> viewport;

    using Clock = std::chrono::steady_clock;
    using Stamp = std::vector<std::pair<U64, U64>>;

    struct Node {
        std::string name;
        std::vector<U64> producers;
        std::vector<U64> reads;
        std::vector<U64> aliases;
        std::vector<U64> consumers;
        std::vector<U64> views;
        bool constant = false;
        bool fixed = false;
        bool settled = false;
        bool throttled = false;
        bool retrying = false;
        bool empty = true;
        U64 sequence = 0;
        U64 frame = 0;
        Stamp stamp;
        Clock::time_point deadline{};
    };

    std::unordered_map<std::string, std::shared_ptr<Module>> modules;
    std::vector<Node> nodes;
    std::vector<U64> sourceNodes;
    std::vector<std::string> presentModules;
    std::unordered_set<std::string> settledModules;
    std::unordered_map<std::string, Module::Timing> settledTimings;
    std::vector<U8> selectedNodes;
    std::vector<Stamp> projectedStamps;
    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> stalledModules;
    Stamp newestStamp;
    Stamp lostStamp;
    U64 stateVersion = 1;
    mutable std::vector<U64> heldVersions;
    mutable std::vector<U8> heldValues;
    mutable std::vector<U64> currentVersions;
    mutable std::vector<U8> currentValues;

    static constexpr std::chrono::milliseconds throttleInterval{100};

    struct RuntimeSegment {
        std::shared_ptr<Runtime> runtime;
        std::vector<U64> nodes;
    };
    std::vector<RuntimeSegment> runtimes;

    Result rebuildOrder();
    Result rebuildRuntimes();
    void touch();
    bool isDue(const Node& node, const Clock::time_point& now) const;
    bool isCurrent(U64 index) const;
    bool isFresh(const Node& node, U64 slot) const;
    bool isAvailable(const Node& node, U64 slot) const;
    bool isHeld(U64 index, const Clock::time_point& now) const;
    const Stamp& stampOf(U64 index) const;
    bool align(U64 index, const Clock::time_point& now, bool& progressed);
    bool hasEmptyInput(const Node& node) const;
    bool bubble(U64 index, bool whole);
    bool isRunnable(U64 index,
                    const std::vector<U8>& readyNodes,
                    const Clock::time_point& now,
                    bool& progressed);
    void project(U64 index);
    bool advance(const RuntimeSegment& segment,
                 const std::unordered_set<std::string>& skippedModules,
                 const std::unordered_set<std::string>& stalledModules,
                 const Clock::time_point& now);
    void settle(Node& node);
    std::chrono::milliseconds idleTimeout(const Clock::time_point& cycleStart);
};

Result SynchronousScheduler::create() {
    JST_DEBUG("[SCHEDULER_SYNCHRONOUS] Creating scheduler.");
    stopping.store(false, std::memory_order_release);
    if (instance) {
        std::shared_ptr<Viewport::Generic> instanceViewport;
        JST_CHECK(instance->viewportGet(instanceViewport));
        viewport = instanceViewport;
    }
    return Result::SUCCESS;
}

Result SynchronousScheduler::destroy() {
    JST_CHECK(lockState([&]{
        JST_DEBUG("[SCHEDULER_SYNCHRONOUS] Destroying scheduler.");

        // Destroy runtimes in reverse order.

        Result result = Result::SUCCESS;
        for (auto& segment : std::ranges::reverse_view(runtimes)) {
            if (segment.runtime) {
                const auto destroyResult = segment.runtime->destroy();
                if (result == Result::SUCCESS && 
                    destroyResult != Result::SUCCESS &&
                    destroyResult != Result::RELOAD) {
                    result = destroyResult;
                }
            }
        }

        runtimes.clear();
        nodes.clear();
        sourceNodes.clear();
        presentModules.clear();
        settledModules.clear();
        settledTimings.clear();
        selectedNodes.clear();
        projectedStamps.clear();
        modules.clear();

        return result;
    }));

    return Result::SUCCESS;
}

Result SynchronousScheduler::start() {
    stopping.store(false, std::memory_order_release);
    clearPresentRequest();

    {
        auto lock = exclusiveSharedLock();
        presentSync = false;
        computeSync = false;
    }

    Scheduler::Context::NotifyPendingCompute();

    return Result::SUCCESS;
}

Result SynchronousScheduler::stop() {
    stopping.store(true, std::memory_order_release);
    clearPresentRequest();
    notifyCompute();
    notifyPresent();
    Scheduler::Context::NotifyPendingCompute();

    return Result::SUCCESS;
}

Result SynchronousScheduler::pause() {
    {
        std::lock_guard lock(handoffMutex);
        pausing.store(true, std::memory_order_release);
    }
    handoffCond.notify_all();

    return Result::SUCCESS;
}

Result SynchronousScheduler::resume() {
    pausing.store(false, std::memory_order_release);
    Scheduler::Context::NotifyPendingCompute();

    return Result::SUCCESS;
}

bool SynchronousScheduler::paused() const {
    return pausing.load(std::memory_order_acquire);
}

Result SynchronousScheduler::add(const std::shared_ptr<Module>& module) {
    if ((module->taint() & Module::Taint::SURFACE) != Module::Taint::CLEAN) {
        JST_CHECK(module->context()->scheduler()->presentInitialize());
    }

    JST_CHECK(lockState([&]{
        if (modules.contains(module->name())) {
            JST_ERROR("[SCHEDULER_SYNCHRONOUS] Module '{}' already exists.", module->name());
            return Result::ERROR;
        }

        modules[module->name()] = module;

        const auto rollback = [&]() -> Result {
            modules.erase(module->name());
            JST_CHECK(this->rebuildOrder());
            return this->rebuildRuntimes();
        };

        auto result = this->rebuildOrder();
        if (result == Result::SUCCESS) {
            result = this->rebuildRuntimes();
        }

        if (result != Result::SUCCESS) {
            const auto rollbackResult = rollback();
            if (rollbackResult != Result::SUCCESS) {
                JST_ERROR("[SCHEDULER_SYNCHRONOUS] Failed to rollback add for module '{}'.", module->name());
                return rollbackResult;
            }

            return result;
        }

        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result SynchronousScheduler::remove(const std::shared_ptr<Module>& module) {
    JST_CHECK(lockState([&]{
        if (!modules.contains(module->name())) {
            return Result::SUCCESS;
        }

        modules.erase(module->name());
        JST_CHECK(this->rebuildOrder());
        JST_CHECK(this->rebuildRuntimes());

        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result SynchronousScheduler::reload(const std::shared_ptr<Module>&) {
    JST_CHECK(lockState([&]{
        JST_CHECK(this->rebuildOrder());
        JST_CHECK(this->rebuildRuntimes());
        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result SynchronousScheduler::synchronize(const std::function<Result()>& fn) {
    if (!fn) {
        return Result::SUCCESS;
    }

    return lockState(fn);
}

Result SynchronousScheduler::present(std::unordered_set<std::string>& failedModules) {
    failedModules.clear();

    if (isStopping() || isPresentHalted() || !presentable.load(std::memory_order_acquire)) {
        return Result::SUCCESS;
    }

    JST_CHECK(capture(failedModules));
    return submit(failedModules);
}

Result SynchronousScheduler::capture(std::unordered_set<std::string>& failedModules) {
    const auto deadline = Clock::now() + kPresentWait;

    while (true) {
        {
            auto dataLock = trySharedDataLock();
            auto lock = dataLock.owns_lock() ? tryExclusiveSharedLock() : std::unique_lock<std::mutex>();

            if (lock.owns_lock() && !isComputeActive()) {
                if (presentModules.empty() || isStopping() || isPresentHalted()) {
                    clearPresentRequest();
                    return Result::SUCCESS;
                }

                setPresentActive();
                clearPresentRequest();

                Result res = Result::SUCCESS;

                for (const auto& name : presentModules) {
                    const auto& mod = modules.at(name);
                    res = mod->context()->scheduler()->presentCapture();
                    if (res != Result::SUCCESS && res != Result::RELOAD) {
                        failedModules.insert(name);
                        break;
                    }
                }

                clearPresentActive();

                if (res != Result::SUCCESS && res != Result::RELOAD) {
                    return res;
                }

                break;
            }
        }

        if (!waitForPresentTurn(deadline)) {
            clearPresentRequest();
            wakePresent();
            return Result::SUCCESS;
        }
    }

    notifyCompute();

    return Result::SUCCESS;
}

Result SynchronousScheduler::submit(std::unordered_set<std::string>& failedModules) {
    std::unique_lock lock(surfaceMutex, std::try_to_lock);
    if (!lock.owns_lock() || isStopping() || isPresentHalted()) {
        return Result::SUCCESS;
    }

    for (const auto& name : presentModules) {
        const auto& mod = modules.at(name);
        const Result res = mod->context()->scheduler()->presentSubmit();
        if (res != Result::SUCCESS && res != Result::RELOAD) {
            failedModules.insert(name);
            return res;
        }
    }

    return Result::SUCCESS;
}

Result SynchronousScheduler::compute(std::unordered_set<std::string>& failedModules) {
    failedModules.clear();

    if (isComputeHalted()) {
        waitComputeHalt();
        return Result::SUCCESS;
    }

    if (isStopping()) {
        clearPresentRequest();
        return Result::SUCCESS;
    }

    if (paused()) {
        return Result::SUCCESS;
    }

    // Phase 1: Snapshot source modules.

    std::vector<std::pair<U64, std::shared_ptr<Module>>> localSourceModules;
    std::vector<U8> readyNodes;
    U64 localGeneration;
    U64 localRuntimeCount;
    {
        auto lock = sharedDataLock();

        localRuntimeCount = runtimes.size();
        if (localRuntimeCount == 0) {
            return Result::SUCCESS;
        }

        if (isComputeHalted()) {
            return Result::SUCCESS;
        }

        localGeneration = currentGeneration();
        readyNodes.assign(nodes.size(), 0);

        for (const auto index : sourceNodes) {
            if (!nodes[index].settled) {
                localSourceModules.emplace_back(index, modules.at(nodes[index].name));
            }
        }
    }

    // Phase 2: Poll source modules (no lock held, must return promptly).

    for (const auto& [index, mod] : localSourceModules) {
        if (isStopping()) {
            clearPresentRequest();
            return Result::SUCCESS;
        }

        if (isComputeHalted()) {
            return Result::SUCCESS;
        }

        if (currentGeneration() != localGeneration || paused()) {
            return Result::SUCCESS;
        }

        const auto& ctx = mod->context();

        if (!ctx || !ctx->scheduler()) {
            readyNodes[index] = 1;
            continue;
        }

        const auto& res = ctx->scheduler()->hasPendingCompute();

        if (isComputeHalted()) {
            return Result::SUCCESS;
        }

        if (currentGeneration() != localGeneration || paused()) {
            return Result::SUCCESS;
        }

        if (res == Result::YIELD || res == Result::TIMEOUT) {
            continue;
        }

        if (res != Result::SUCCESS && res != Result::RELOAD) {
            failedModules.insert(mod->name());
            return res;
        }

        readyNodes[index] = 1;
    }

    // Phase 3: Execute runnable modules with priority yield mechanism.

    Result res = Result::SUCCESS;
    skippedModules.clear();
    stalledModules.clear();
    bool progressed = false;
    bool started = false;
    const auto now = Clock::now();

    {
        auto lock = sharedDataLock();
        touch();
    }

    for (U64 i = 0; i < localRuntimeCount; i++) {
        std::vector<std::string> pendingModules;

        // Priority Yield: Check if present wants in before each segment.

        if (isPresentRequested()) {
            yieldToPresent(started);

            if (isStopping()) {
                return Result::SUCCESS;
            }

            if (isComputeHalted() || (!started && paused())) {
                return Result::SUCCESS;
            }
        }

        {
            std::unique_lock dataLock(dataMutex);

            if (currentGeneration() != localGeneration || (!started && paused())) {
                return Result::SUCCESS;
            }

            if (isComputeHalted() || nodes.empty()) {
                return Result::SUCCESS;
            }

            const auto& segment = runtimes[i];
            pendingModules.reserve(segment.nodes.size());

            for (const auto index : segment.nodes) {
                if (isRunnable(index, readyNodes, now, progressed)) {
                    selectedNodes[index] = 1;
                    project(index);
                    pendingModules.push_back(nodes[index].name);
                }
            }

            if (pendingModules.empty()) {
                continue;
            }

            const auto clearSelection = [&]() {
                for (const auto index : segment.nodes) {
                    selectedNodes[index] = 0;
                }
            };

            auto lock = exclusiveSharedLock();
            waitForPresentToFinish(lock);

            if (isStopping()) {
                clearSelection();
                clearPresentRequest();
                return Result::SUCCESS;
            }

            if (isComputeHalted()) {
                clearSelection();
                return Result::SUCCESS;
            }

            setComputeActive();

            started = true;
            res = segment.runtime->compute(pendingModules, skippedModules, stalledModules, failedModules);

            if (res == Result::SUCCESS) {
                progressed |= advance(segment, skippedModules, stalledModules, now);
            }

            clearSelection();
            clearComputeActive();
        }

        notifyPresent();

        if (res != Result::SUCCESS) {
            if (res == Result::ERROR && failedModules.empty()) {
                for (const auto& name : pendingModules) {
                    failedModules.insert(name);
                }
            }
            break;
        }
    }

    if (progressed) {
        ReportComputeProgress();
    }

    if (res != Result::SUCCESS || progressed) {
        return res;
    }

    if (isStopping()) {
        clearPresentRequest();
        return Result::SUCCESS;
    }

    ReportComputeDeadline(Clock::now() + idleTimeout(now));

    return Result::SUCCESS;
}

std::shared_ptr<Scheduler::Impl> SynchronousSchedulerFactory() {
    return std::make_shared<SynchronousScheduler>();
}

Result SynchronousScheduler::rebuildOrder() {
    std::unordered_map<std::string, Clock::time_point> deadlines;
    for (const auto& node : nodes) {
        deadlines[node.name] = node.deadline;
    }

    nodes.clear();
    sourceNodes.clear();
    presentModules.clear();
    selectedNodes.clear();
    projectedStamps.clear();

    // Build adjacency lists and calculate in-degrees for topological sort.

    std::unordered_map<std::string, std::vector<std::string>> consumerNames;
    std::unordered_map<std::string, size_t> inDegree;

    for (const auto& [name, _] : modules) {
        inDegree[name] = 0;
        consumerNames[name];
    }

    for (const auto& [consumerName, mod] : modules) {
        for (const auto& [slot, link] : mod->inputs()) {
            (void)slot;

            if (!link.producer.has_value() || !modules.contains(link.producer->module)) {
                continue;
            }

            consumerNames[link.producer->module].push_back(consumerName);
            inDegree[consumerName] += 1;
        }

        if ((mod->taint() & Module::Taint::SURFACE) != Module::Taint::CLEAN) {
            presentModules.push_back(consumerName);
        }
    }

    // Kahn's algorithm for topological sort.

    std::queue<std::string> q;
    std::vector<std::string> order;

    for (const auto& [name, deg] : inDegree) {
        if (deg == 0) {
            q.push(name);
        }
    }

    while (!q.empty()) {
        auto u = q.front();
        q.pop();
        order.push_back(u);

        for (const auto& v : consumerNames[u]) {
            if (--inDegree[v] == 0) {
                q.push(v);
            }
        }
    }

    if (order.size() != modules.size()) {
        JST_ERROR("[SCHEDULER_SYNCHRONOUS] Detected cycle or unresolved dependencies in the module DAG.");
        presentModules.clear();
        return Result::ERROR;
    }

    // Lay nodes out in topological order and resolve their connections.

    std::unordered_map<std::string, U64> indices;
    nodes.resize(order.size());

    for (U64 index = 0; index < order.size(); ++index) {
        indices[order[index]] = index;
        nodes[index].name = order[index];
    }

    for (U64 index = 0; index < order.size(); ++index) {
        auto& node = nodes[index];
        const auto& mod = modules.at(node.name);

        for (const auto& [slot, link] : mod->inputs()) {
            (void)slot;

            if (!link.producer.has_value() || !modules.contains(link.producer->module)) {
                continue;
            }

            const U64 producer = indices.at(link.producer->module);
            const bool aliasesInput = std::ranges::any_of(mod->outputs(), [&tensor = link.tensor](const auto& entry) {
                return entry.second.tensor.sharesStorage(tensor) ||
                       entry.second.tensor.inheritsAttributes(tensor) ||
                       entry.second.tensor.derivesAttributes();
            });

            if (aliasesInput) {
                node.aliases.push_back(node.producers.size());

                auto& views = nodes[producer].views;
                if (std::ranges::find(views, index) == views.end()) {
                    views.push_back(index);
                }
            }

            node.producers.push_back(producer);
            node.reads.push_back(0);

            auto& consumers = nodes[producer].consumers;
            if (std::ranges::find(consumers, index) == consumers.end()) {
                consumers.push_back(index);
            }
        }

        if (node.producers.empty()) {
            sourceNodes.push_back(index);
            node.constant = (mod->taint() & Module::Taint::STATIC_OUTPUT) != Module::Taint::CLEAN;
        }

        node.throttled = (mod->taint() & Module::Taint::THROTTLED) != Module::Taint::CLEAN;
        node.settled = settledModules.contains(node.name);

        if (node.throttled && deadlines.contains(node.name)) {
            node.deadline = deadlines.at(node.name);
        }
    }

    selectedNodes.assign(nodes.size(), 0);
    projectedStamps.resize(nodes.size());
    heldVersions.assign(nodes.size(), 0);
    heldValues.assign(nodes.size(), 0);
    currentVersions.assign(nodes.size(), 0);
    currentValues.assign(nodes.size(), 0);
    touch();

    // Input mutation can invalidate retained or aliased static buffers. Disable
    // settling until the scheduler has enough alias information to prove safety.

    const bool hasInPlaceModule = std::ranges::any_of(modules, [](const auto& entry) {
        return (entry.second->taint() & Module::Taint::IN_PLACE) != Module::Taint::CLEAN;
    });

    if (hasInPlaceModule) {
        return Result::SUCCESS;
    }

    // Derive static descendants from explicit roots. STATELESS modules only inherit
    // staticness when every input has a known static producer in this graph.

    for (auto& node : nodes) {
        const auto& mod = modules.at(node.name);

        if ((mod->taint() & Module::Taint::STATIC_OUTPUT) != Module::Taint::CLEAN) {
            node.fixed = true;
            continue;
        }

        if ((mod->taint() & Module::Taint::STATELESS) == Module::Taint::CLEAN ||
            mod->inputs().empty()) {
            continue;
        }

        node.fixed = std::ranges::all_of(mod->inputs(), [&](const auto& entry) {
            const auto& link = entry.second;
            return link.producer.has_value() &&
                   indices.contains(link.producer->module) &&
                   nodes[indices.at(link.producer->module)].fixed;
        });
    }

    return Result::SUCCESS;
}

Result SynchronousScheduler::rebuildRuntimes() {
    for (auto& segment : std::ranges::reverse_view(runtimes)) {
        if (segment.runtime) {
            segment.runtime->destroy();
        }
    }
    runtimes.clear();

    if (nodes.empty()) {
        return Result::SUCCESS;
    }

    DeviceType currentDevice = modules.at(nodes.front().name)->device();
    RuntimeType currentRuntime = modules.at(nodes.front().name)->runtime();
    Runtime::Modules segmentModules;
    std::vector<U64> segmentNodes;

    auto flushSegment = [&]() -> Result {
        if (segmentModules.empty()) {
            return Result::SUCCESS;
        }

        const std::string runtimeName = jst::fmt::format("{}", runtimes.size());
        auto runtime = std::make_shared<Runtime>(runtimeName, currentDevice, currentRuntime);
        JST_CHECK(runtime->create(segmentModules));

        runtimes.push_back({std::move(runtime), std::move(segmentNodes)});
        segmentModules.clear();
        segmentNodes.clear();

        return Result::SUCCESS;
    };

    for (U64 index = 0; index < nodes.size(); ++index) {
        const auto& mod = modules.at(nodes[index].name);
        const DeviceType dev = mod->device();
        const RuntimeType run = mod->runtime();

        if (dev != currentDevice || run != currentRuntime) {
            JST_CHECK(flushSegment());
            currentDevice = dev;
            currentRuntime = run;
        }

        segmentModules[nodes[index].name] = mod;
        segmentNodes.push_back(index);
    }

    JST_CHECK(flushSegment());

    return Result::SUCCESS;
}

void SynchronousScheduler::touch() {
    stateVersion += 1;
}

bool SynchronousScheduler::isDue(const Node& node, const Clock::time_point& now) const {
    return !node.throttled || now >= node.deadline;
}

bool SynchronousScheduler::isCurrent(const U64 index) const {
    if (currentVersions[index] == stateVersion) {
        return currentValues[index];
    }

    const auto& node = nodes[index];
    const bool current = !node.empty && std::ranges::all_of(node.aliases, [&](const auto slot) {
        const U64 producer = node.producers[slot];
        return isCurrent(producer) && node.reads[slot] == nodes[producer].sequence;
    });

    currentVersions[index] = stateVersion;
    currentValues[index] = current;
    return current;
}

bool SynchronousScheduler::isFresh(const Node& node, const U64 slot) const {
    const U64 producer = node.producers[slot];
    return node.reads[slot] != nodes[producer].sequence &&
           (nodes[producer].empty || isCurrent(producer));
}

bool SynchronousScheduler::isAvailable(const Node& node, const U64 slot) const {
    const U64 producer = node.producers[slot];
    return nodes[producer].settled || selectedNodes[producer] || isFresh(node, slot);
}

bool SynchronousScheduler::isHeld(const U64 index, const Clock::time_point& now) const {
    if (heldVersions[index] == stateVersion) {
        return heldValues[index];
    }

    const auto direct = [&]() {
        for (const auto consumer : nodes[index].consumers) {
            const auto& node = nodes[consumer];

            if (node.settled) {
                continue;
            }

            for (U64 slot = 0; slot < node.producers.size(); ++slot) {
                if (node.producers[slot] == index && isFresh(node, slot)) {
                    return true;
                }
            }
        }

        return false;
    };

    const bool held = direct() || std::ranges::any_of(nodes[index].views, [&](const auto view) {
        return isHeld(view, now);
    });

    heldVersions[index] = stateVersion;
    heldValues[index] = held;
    return held;
}

const SynchronousScheduler::Stamp& SynchronousScheduler::stampOf(const U64 index) const {
    return selectedNodes[index] ? projectedStamps[index] : nodes[index].stamp;
}

bool SynchronousScheduler::align(const U64 index, const Clock::time_point& now, bool& progressed) {
    auto& node = nodes[index];

    if (node.producers.size() < 2 || isHeld(index, now)) {
        return node.producers.size() < 2;
    }

    const auto raise = [](Stamp& stamp, const U64 source, const U64 frame) {
        const auto entry = std::ranges::find(stamp, source, &Stamp::value_type::first);
        if (entry == stamp.end()) {
            stamp.emplace_back(source, frame);
        } else {
            entry->second = std::max(entry->second, frame);
        }
    };

    lostStamp.clear();
    bool discarded = false;

    while (true) {
        newestStamp.clear();

        for (U64 slot = 0; slot < node.producers.size(); ++slot) {
            const U64 producer = node.producers[slot];
            const bool available = isAvailable(node, slot);

            for (const auto& [source, frame] : stampOf(producer)) {
                raise(newestStamp, source, available ? frame : frame + 1);
            }
        }

        bool aligned = true;
        bool changed = false;

        for (U64 slot = 0; slot < node.producers.size(); ++slot) {
            if (!isAvailable(node, slot)) {
                continue;
            }

            const U64 producer = node.producers[slot];
            const bool behind = std::ranges::any_of(stampOf(producer), [&](const auto& entry) {
                return std::ranges::find(newestStamp, entry.first, &Stamp::value_type::first)->second > entry.second;
            });

            if (!behind) {
                continue;
            }

            aligned = false;

            if (!nodes[producer].settled && !selectedNodes[producer]) {
                for (const auto& [source, frame] : nodes[producer].stamp) {
                    raise(lostStamp, source, frame);
                }
                node.reads[slot] = nodes[producer].sequence;
                discarded = true;
                changed = true;
                touch();
            }
        }

        if (changed) {
            continue;
        }

        if (!discarded) {
            return aligned;
        }

        node.stamp = lostStamp;
        node.sequence += 1;
        node.empty = true;
        node.retrying = false;
        progressed = true;
        touch();

        return false;
    }
}

bool SynchronousScheduler::hasEmptyInput(const Node& node) const {
    for (U64 slot = 0; slot < node.producers.size(); ++slot) {
        const U64 producer = node.producers[slot];
        if (!selectedNodes[producer] && nodes[producer].empty && isFresh(node, slot)) {
            return true;
        }
    }

    return false;
}

bool SynchronousScheduler::bubble(const U64 index, const bool whole) {
    auto& node = nodes[index];

    bool lost = whole;

    for (U64 slot = 0; slot < node.producers.size(); ++slot) {
        const U64 producer = node.producers[slot];
        if (nodes[producer].settled) {
            continue;
        }

        if (selectedNodes[producer] || !isFresh(node, slot)) {
            return false;
        }

        if (!whole && nodes[producer].empty && isFresh(node, slot)) {
            lost = true;
        }
    }

    if (!lost) {
        return false;
    }

    auto& stamp = projectedStamps[index];
    stamp.clear();

    lost = false;

    for (U64 slot = 0; slot < node.producers.size(); ++slot) {
        const U64 producer = node.producers[slot];
        if (nodes[producer].settled || !isFresh(node, slot)) {
            continue;
        }

        for (const auto& [source, frame] : nodes[producer].stamp) {
            const auto entry = std::ranges::find(stamp, source, &Stamp::value_type::first);
            if (entry == stamp.end()) {
                stamp.emplace_back(source, frame);
            } else {
                entry->second = std::max(entry->second, frame);
            }
        }

        node.reads[slot] = nodes[producer].sequence;
        lost = true;
    }

    if (!lost) {
        return false;
    }

    node.stamp = stamp;
    node.sequence += 1;
    node.empty = true;
    node.retrying = false;
    touch();

    return true;
}

bool SynchronousScheduler::isRunnable(const U64 index,
                                      const std::vector<U8>& readyNodes,
                                      const Clock::time_point& now,
                                      bool& progressed) {
    const auto& node = nodes[index];

    if (node.settled || isHeld(index, now)) {
        return false;
    }

    if (node.producers.empty()) {
        return isDue(node, now) && readyNodes[index];
    }

    if (!isDue(node, now)) {
        if (!node.retrying) {
            progressed |= bubble(index, true);
        }
        return false;
    }

    if (hasEmptyInput(node) && bubble(index, false)) {
        progressed = true;
        return false;
    }

    const bool aligned = align(index, now, progressed);

    if (isHeld(index, now)) {
        return false;
    }

    for (U64 slot = 0; slot < node.producers.size(); ++slot) {
        if (!isAvailable(node, slot)) {
            return false;
        }

        const U64 producer = node.producers[slot];
        if (!selectedNodes[producer] && !nodes[producer].settled && nodes[producer].empty) {
            return false;
        }
    }

    return aligned;
}

void SynchronousScheduler::project(const U64 index) {
    const auto& node = nodes[index];
    auto& stamp = projectedStamps[index];
    stamp.clear();

    if (node.fixed) {
        return;
    }

    if (node.producers.empty()) {
        if (!node.constant) {
            stamp.emplace_back(index, node.frame + 1);
        }
        return;
    }

    for (const auto producer : node.producers) {
        for (const auto& [source, frame] : stampOf(producer)) {
            const auto entry = std::ranges::find(stamp, source, &Stamp::value_type::first);
            if (entry == stamp.end()) {
                stamp.emplace_back(source, frame);
            } else {
                entry->second = std::max(entry->second, frame);
            }
        }
    }

    if (stamp.empty() && !node.fixed) {
        stamp.emplace_back(index, node.frame + 1);
    }
}

bool SynchronousScheduler::advance(const RuntimeSegment& segment,
                                   const std::unordered_set<std::string>& skippedModules,
                                   const std::unordered_set<std::string>& stalledModules,
                                   const Clock::time_point& now) {
    bool progressed = false;
    const auto submitted = Clock::now();

    for (const auto index : segment.nodes) {
        auto& node = nodes[index];

        if (!selectedNodes[index]) {
            if (node.settled) {
                continue;
            }

            if (isHeld(index, now)) {
                continue;
            }

            if (!isDue(node, now)) {
                if (!node.retrying) {
                    progressed |= bubble(index, true);
                }
            } else if (hasEmptyInput(node) && bubble(index, false)) {
                progressed = true;
            } else {
                align(index, now, progressed);
            }
            continue;
        }

        selectedNodes[index] = 0;
        touch();

        if (stalledModules.contains(node.name)) {
            const bool attempted = std::ranges::none_of(node.producers, [&](const auto producer) {
                return skippedModules.contains(nodes[producer].name) || stalledModules.contains(nodes[producer].name);
            });

            if (attempted) {
                node.retrying = true;
                if (node.throttled) {
                    node.deadline = submitted + throttleInterval;
                }
            }

            progressed |= bubble(index, false);
            continue;
        }

        progressed = true;

        if (node.throttled) {
            node.deadline = submitted + throttleInterval;
        }

        for (U64 slot = 0; slot < node.producers.size(); ++slot) {
            node.reads[slot] = nodes[node.producers[slot]].sequence;
        }

        node.frame += 1;
        node.stamp = projectedStamps[index];
        node.sequence += 1;
        node.retrying = false;
        node.empty = skippedModules.contains(node.name);

        if (!node.empty && node.fixed) {
            settle(node);
        }
    }

    return progressed;
}

void SynchronousScheduler::settle(Node& node) {
    node.settled = true;
    touch();

    if (!settledModules.insert(node.name).second) {
        return;
    }

    const auto& mod = modules.at(node.name);
    auto timing = mod->timing();
    settledTimings[node.name] = timing;
    timing.computeTime = 0.0f;
    mod->timing(timing);
}

std::chrono::milliseconds SynchronousScheduler::idleTimeout(const Clock::time_point& cycleStart) {
    auto lock = sharedDataLock();

    const auto now = Clock::now();
    auto deadline = now + throttleInterval;

    for (const auto& node : nodes) {
        if (node.throttled && !node.settled && node.deadline > cycleStart) {
            deadline = std::min(deadline, node.deadline);
        }
    }

    if (deadline <= now) {
        return std::chrono::milliseconds(0);
    }

    return std::chrono::ceil<std::chrono::milliseconds>(deadline - now);
}

void SynchronousScheduler::haltAll() {
    computeHalt.test_and_set();
    presentHalt.test_and_set();
}

void SynchronousScheduler::resumeAll() {
    computeHalt.clear();
    computeHalt.notify_all();
    presentHalt.clear();
    presentHalt.notify_all();
}

bool SynchronousScheduler::isComputeHalted() const {
    return computeHalt.test();
}

bool SynchronousScheduler::isPresentHalted() const {
    return presentHalt.test();
}

void SynchronousScheduler::waitComputeHalt() {
    computeHalt.wait(true);
}

U64 SynchronousScheduler::currentGeneration() const {
    return generation.load(std::memory_order_acquire);
}

void SynchronousScheduler::incrementGeneration() {
    generation.fetch_add(1, std::memory_order_release);
}

void SynchronousScheduler::requestPresentEntry() {
    std::lock_guard lock(handoffMutex);
    presentWantsIn.store(true, std::memory_order_release);
}

void SynchronousScheduler::clearPresentRequest() {
    {
        std::lock_guard lock(handoffMutex);
        presentWantsIn.store(false, std::memory_order_release);
    }
    handoffCond.notify_all();
}

bool SynchronousScheduler::isPresentRequested() const {
    return presentWantsIn.load(std::memory_order_acquire);
}

bool SynchronousScheduler::isStopping() const {
    return stopping.load(std::memory_order_acquire);
}

void SynchronousScheduler::setComputeActive() {
    computeSync = true;
}

void SynchronousScheduler::clearComputeActive() {
    computeSync = false;
}

bool SynchronousScheduler::isComputeActive() const {
    return computeSync;
}

void SynchronousScheduler::setPresentActive() {
    presentSync = true;
}

void SynchronousScheduler::clearPresentActive() {
    presentSync = false;
}

std::shared_lock<std::shared_mutex> SynchronousScheduler::sharedDataLock() {
    return std::shared_lock(dataMutex);
}

std::shared_lock<std::shared_mutex> SynchronousScheduler::trySharedDataLock() {
    return std::shared_lock(dataMutex, std::try_to_lock);
}

std::unique_lock<std::mutex> SynchronousScheduler::exclusiveSharedLock() {
    return std::unique_lock(sharedMutex);
}

std::unique_lock<std::mutex> SynchronousScheduler::tryExclusiveSharedLock() {
    return std::unique_lock(sharedMutex, std::try_to_lock);
}

void SynchronousScheduler::waitForPresentToFinish(std::unique_lock<std::mutex>& lock) {
    computeCond.wait(lock, [&] {
        return !presentSync || computeHalt.test() || isStopping();
    });
}

bool SynchronousScheduler::waitForPresentTurn(const Clock::time_point deadline) {
    if (isStopping() || isPresentHalted() || Clock::now() >= deadline) {
        return false;
    }

    requestPresentEntry();

    std::unique_lock lock(handoffMutex);
    presentCond.wait_until(lock, std::min(deadline, Clock::now() + kPresentPoll));
    return !isStopping() && !isPresentHalted();
}

void SynchronousScheduler::yieldToPresent(const bool started) {
    std::unique_lock lock(handoffMutex);
    handoffCond.wait(lock, [&] {
        return !isPresentRequested() || isStopping() || isComputeHalted() || (!started && paused());
    });
}

void SynchronousScheduler::notifyCompute() {
    computeCond.notify_all();
}

void SynchronousScheduler::notifyPresent() {
    {
        std::lock_guard lock(handoffMutex);
    }
    presentCond.notify_all();
}

void SynchronousScheduler::wakePresent() {
    if (const auto target = viewport.lock()) {
        target->input().wake();
    }
}

Result SynchronousScheduler::lockState(const std::function<Result()>& func) {
    haltAll();

    std::unique_lock dataLock(dataMutex);
    std::unique_lock surfaceLock(surfaceMutex);

    sharedMutex.lock();
    presentSync = true;
    computeSync = true;

    for (const auto& [name, timing] : settledTimings) {
        if (!modules.contains(name)) {
            continue;
        }

        modules.at(name)->timing(timing);
    }

    settledTimings.clear();
    settledModules.clear();

    for (auto& node : nodes) {
        node.settled = false;
        node.retrying = false;
        for (U64 slot = 0; slot < node.producers.size(); ++slot) {
            node.reads[slot] = nodes[node.producers[slot]].sequence;
        }
    }
    touch();

    Result res = func();

    presentable.store(!presentModules.empty(), std::memory_order_release);
    incrementGeneration();
    clearPresentRequest();
    computeSync = false;
    presentSync = false;
    sharedMutex.unlock();
    surfaceLock.unlock();
    dataLock.unlock();

    notifyCompute();
    notifyPresent();

    resumeAll();

    Scheduler::Context::NotifyPendingCompute();

    return res;
}

}  // namespace Jetstream
