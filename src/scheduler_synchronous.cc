// Synchronous Scheduler Implementation
//
// Coordinates compute and present threads with the following guarantees:
//   - Present never blocks waiting for compute (skips frame if busy).
//   - Present has priority: compute yields at segment boundaries when present wants in.
//   - State modifications (add/remove/reload) safely halt both threads.
//   - Compute polling (hasPendingCompute) doesn't hold locks, allowing fast state changes.

#include <jetstream/detail/scheduler_impl.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/runtime.hh>
#include <jetstream/module.hh>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <ranges>
#include <shared_mutex>
#include <thread>

namespace Jetstream {

struct SynchronousScheduler : public Scheduler::Impl {
 public:
    Result create() override;
    Result destroy() override;
    Result start() override;
    Result stop() override;

    Result add(const std::shared_ptr<Module>& module) override;
    Result remove(const std::shared_ptr<Module>& module) override;
    Result reload(const std::shared_ptr<Module>& module) override;

    Result present() override;
    Result compute() override;

 private:
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
    void notifyCompute();
    void notifyPresent();
    Result lockState(const std::function<Result()>& func);

    std::atomic_flag computeHalt = ATOMIC_FLAG_INIT;
    std::atomic_flag presentHalt = ATOMIC_FLAG_INIT;
    std::shared_mutex dataMutex;
    std::mutex sharedMutex;
    std::condition_variable presentCond;
    std::condition_variable computeCond;
    bool presentSync = false;
    bool computeSync = false;
    std::atomic<bool> presentWantsIn{false};
    std::atomic<bool> stopping{false};
    std::atomic<U64> generation{0};

    std::unordered_map<std::string, std::shared_ptr<Module>> modules;
    std::vector<std::string> topoOrder;
    std::vector<std::string> sourceModules;
    std::vector<std::string> presentModules;

    struct RuntimeSegment {
        std::shared_ptr<Runtime> runtime;
        std::vector<std::string> modules;
    };
    std::vector<RuntimeSegment> runtimes;
    std::unordered_map<std::string, std::shared_ptr<Runtime::Metrics>> metrics_;

    Result rebuildOrder();
    Result rebuildRuntimes();

    const std::unordered_map<std::string, std::shared_ptr<Runtime::Metrics>>& metrics() const override;
};

Result SynchronousScheduler::create() {
    JST_DEBUG("[SCHEDULER_SYNCHRONOUS] Creating scheduler.");
    stopping.store(false, std::memory_order_release);
    return Result::SUCCESS;
}

Result SynchronousScheduler::destroy() {
    JST_CHECK(lockState([&]{
        JST_DEBUG("[SCHEDULER_SYNCHRONOUS] Destroying scheduler.");

        // Destroy runtimes in reverse order.

        for (auto& segment : std::ranges::reverse_view(runtimes)) {
            if (segment.runtime) {
                segment.runtime->destroy();
            }
        }

        runtimes.clear();
        topoOrder.clear();
        sourceModules.clear();
        modules.clear();

        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result SynchronousScheduler::start() {
    stopping.store(false, std::memory_order_release);
    clearPresentRequest();

    auto lock = exclusiveSharedLock();
    presentSync = false;
    computeSync = false;

    return Result::SUCCESS;
}

Result SynchronousScheduler::stop() {
    stopping.store(true, std::memory_order_release);
    clearPresentRequest();
    notifyCompute();
    notifyPresent();

    return Result::SUCCESS;
}

Result SynchronousScheduler::add(const std::shared_ptr<Module>& module) {
    JST_CHECK(module->context()->scheduler()->presentInitialize());

    JST_CHECK(lockState([&]{
        if (modules.contains(module->name())) {
            JST_ERROR("[SCHEDULER_SYNCHRONOUS] Module '{}' already exists.", module->name());
            return Result::ERROR;
        }

        modules[module->name()] = module;
        JST_CHECK(this->rebuildOrder());
        JST_CHECK(this->rebuildRuntimes());

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

Result SynchronousScheduler::present() {
    if (isStopping() || isPresentHalted()) {
        clearPresentRequest();
        return Result::SUCCESS;
    }

    {
        auto dataLock = trySharedDataLock();
        if (!dataLock.owns_lock()) {
            requestPresentEntry();
            return Result::SUCCESS;
        }

        if (presentModules.empty()) {
            clearPresentRequest();
            return Result::SUCCESS;
        }

        if (isStopping() || isPresentHalted()) {
            clearPresentRequest();
            return Result::SUCCESS;
        }

        auto lock = tryExclusiveSharedLock();
        if (!lock.owns_lock()) {
            if (isStopping()) {
                clearPresentRequest();
                return Result::SUCCESS;
            }

            requestPresentEntry();
            return Result::SUCCESS;
        }

        if (isStopping()) {
            clearPresentRequest();
            return Result::SUCCESS;
        }

        if (isComputeActive()) {
            if (isStopping()) {
                clearPresentRequest();
                return Result::SUCCESS;
            }

            requestPresentEntry();
            return Result::SUCCESS;
        }

        setPresentActive();
        clearPresentRequest();

        Result res = Result::SUCCESS;

        for (const auto& name : presentModules) {
            const auto& mod = modules.at(name);
            res = mod->context()->scheduler()->presentSubmit();
            if (res != Result::SUCCESS && res != Result::RELOAD) {
                break;
            }
        }

        clearPresentActive();

        if (res != Result::SUCCESS && res != Result::RELOAD) {
            return res;
        }
    }

    notifyCompute();

    return Result::SUCCESS;
}

Result SynchronousScheduler::compute() {
    if (isComputeHalted()) {
        waitComputeHalt();
        return Result::SUCCESS;
    }

    if (isStopping()) {
        clearPresentRequest();
        return Result::SUCCESS;
    }

    // Phase 1: Snapshot source modules.

    std::vector<std::shared_ptr<Module>> localSourceModules;
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

        for (const auto& name : sourceModules) {
            if (modules.contains(name)) {
                localSourceModules.push_back(modules.at(name));
            }
        }
    }

    // Phase 2: Poll source modules (no lock held, can block).

    {
        while (true) {
            if (isStopping()) {
                clearPresentRequest();
                return Result::SUCCESS;
            }

            if (currentGeneration() != localGeneration) {
                return Result::SUCCESS;
            }

            bool allReady = true;

            for (const auto& mod : localSourceModules) {
                if (isComputeHalted()) {
                    return Result::SUCCESS;
                }

                if (currentGeneration() != localGeneration) {
                    return Result::SUCCESS;
                }

                const auto& ctx = mod->context();

                if (!ctx) {
                    continue;
                }

                const auto& schedulerCtx = ctx->scheduler();

                if (!schedulerCtx) {
                    continue;
                }

                const auto& res = schedulerCtx->hasPendingCompute();

                if (isComputeHalted()) {
                    return Result::SUCCESS;
                }

                if (currentGeneration() != localGeneration) {
                    return Result::SUCCESS;
                }

                if (res == Result::YIELD || res == Result::TIMEOUT) {
                    allReady = false;
                    break;
                }

                JST_CHECK(res);
            }

            if (allReady) {
                break;
            }
        }
    }

    // Phase 3: Execute modules with priority yield mechanism.

    Result res = Result::SUCCESS;
    std::unordered_set<std::string> skippedModules;

    for (U64 i = 0; i < localRuntimeCount; i++) {
        // Priority Yield: Check if present wants in before each segment.

        if (isPresentRequested()) {
            while (isPresentRequested()) {
                if (isStopping()) {
                    clearPresentRequest();
                    return Result::SUCCESS;
                }

                if (isComputeHalted()) {
                    return Result::SUCCESS;
                }
                std::this_thread::yield();
            }

            if (isComputeHalted()) {
                return Result::SUCCESS;
            }
        }

        {
            auto dataLock = sharedDataLock();

            if (currentGeneration() != localGeneration) {
                return Result::SUCCESS;
            }

            if (isComputeHalted() || topoOrder.empty()) {
                return Result::SUCCESS;
            }

            auto lock = exclusiveSharedLock();
            waitForPresentToFinish(lock);

            if (isStopping()) {
                clearPresentRequest();
                return Result::SUCCESS;
            }

            if (isComputeHalted()) {
                return Result::SUCCESS;
            }

            setComputeActive();

            res = runtimes[i].runtime->compute(runtimes[i].modules, skippedModules);

            clearComputeActive();
        }

        notifyPresent();

        if (res != Result::SUCCESS && res != Result::YIELD) {
            break;
        }
    }

    if (res == Result::YIELD || res == Result::TIMEOUT) {
        return Result::SUCCESS;
    }

    return res;
}

std::shared_ptr<Scheduler::Impl> SynchronousSchedulerFactory() {
    return std::make_shared<SynchronousScheduler>();
}

Result SynchronousScheduler::rebuildOrder() {
    topoOrder.clear();
    sourceModules.clear();
    presentModules.clear();

    // Build adjacency list and calculate in-degrees for topological sort.

    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, size_t> inDegree;

    for (const auto& [name, _] : modules) {
        inDegree[name] = 0;
        adj[name];
    }

    for (const auto& [consumerName, mod] : modules) {
        const auto& inputs = mod->inputs();

        for (const auto& [slot, link] : inputs) {
            (void)slot;

            if (!link.producer.has_value()) {
                continue;
            }

            if (!modules.contains(link.producer->module)) {
                continue;
            }

            adj[link.producer->module].push_back(consumerName);
            inDegree[consumerName] += 1;
        }
    }

    // Kahn's algorithm for topological sort.

    std::queue<std::string> q;

    for (const auto& [name, deg] : inDegree) {
        if (deg == 0) {
            q.push(name);
            sourceModules.push_back(name);
        }

        if (modules.at(name)->surface()) {
            presentModules.push_back(name);
        }
    }

    while (!q.empty()) {
        auto u = q.front();
        q.pop();
        topoOrder.push_back(u);

        for (const auto& v : adj[u]) {
            if (--inDegree[v] == 0) {
                q.push(v);
            }
        }
    }

    if (topoOrder.size() != modules.size()) {
        JST_ERROR("[SCHEDULER_SYNCHRONOUS] Detected cycle or unresolved dependencies in the module DAG.");
        topoOrder.clear();
        return Result::ERROR;
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
    metrics_.clear();

    if (topoOrder.empty()) {
        return Result::SUCCESS;
    }

    DeviceType currentDevice = modules.at(topoOrder.front())->device();
    RuntimeType currentRuntime = modules.at(topoOrder.front())->runtime();
    Runtime::Modules segmentModules;
    std::vector<std::string> segmentNames;

    auto flushSegment = [&]() -> Result {
        if (segmentModules.empty()) {
            return Result::SUCCESS;
        }

        const std::string runtimeName = jst::fmt::format("{}", runtimes.size());
        auto runtime = std::make_shared<Runtime>(runtimeName, currentDevice, currentRuntime);
        JST_CHECK(runtime->create(segmentModules));

        // Assign metrics for each block.

        std::unordered_set<std::string> blocks;

        for (const auto& fullName : segmentNames) {
            blocks.insert(Parser::SplitString(fullName, "-")[0]);
        }

        for (const auto& name : blocks) {
            metrics_[name] = runtime->metrics();
        }

        runtimes.push_back({std::move(runtime), std::move(segmentNames)});
        segmentModules.clear();
        segmentNames.clear();

        return Result::SUCCESS;
    };

    for (const auto& name : topoOrder) {
        const auto& mod = modules.at(name);
        const DeviceType dev = mod->device();
        const RuntimeType run = mod->runtime();

        if (dev != currentDevice || run != currentRuntime) {
            JST_CHECK(flushSegment());
            currentDevice = dev;
            currentRuntime = run;
        }

        segmentModules[name] = mod;
        segmentNames.push_back(name);
    }

    JST_CHECK(flushSegment());

    return Result::SUCCESS;
}

const std::unordered_map<std::string, std::shared_ptr<Runtime::Metrics>>& SynchronousScheduler::metrics() const {
    return metrics_;
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
    presentWantsIn.store(true, std::memory_order_release);
}

void SynchronousScheduler::clearPresentRequest() {
    presentWantsIn.store(false, std::memory_order_release);
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

void SynchronousScheduler::notifyCompute() {
    computeCond.notify_all();
}

void SynchronousScheduler::notifyPresent() {
    presentCond.notify_all();
}

Result SynchronousScheduler::lockState(const std::function<Result()>& func) {
    haltAll();

    std::unique_lock dataLock(dataMutex);

    sharedMutex.lock();
    presentSync = true;
    computeSync = true;

    Result res = func();

    incrementGeneration();
    clearPresentRequest();
    computeSync = false;
    presentSync = false;
    sharedMutex.unlock();
    dataLock.unlock();

    notifyCompute();
    notifyPresent();

    resumeAll();

    return res;
}

}  // namespace Jetstream
