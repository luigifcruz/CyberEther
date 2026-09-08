#include "runtime/python/dependencies/coordinator.hh"

#include <algorithm>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include "jetstream/runtime_context_python.hh"
#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/bootstrap/base.hh"
#include "runtime/python/dependencies/base.hh"

namespace Jetstream {

namespace {

Result InstallPythonDependencies(U64 generation);

struct PythonDependencyRecord {
    std::vector<std::string> requirements;
    std::string block;
    std::weak_ptr<Flowgraph::View> view;
    bool scheduled = false;
};

struct PythonDependencyRegistry {
    std::mutex mutex;
    std::unordered_map<PythonRuntimeContext*, PythonDependencyRecord> records;
    PythonDependencyEnvironment activeEnvironment;
    PythonDependencyRequest request;
    bool hasResolvedDependencies = false;
    std::optional<PythonDependencyPolicy> dependencyPolicy;
    U64 resolvedGeneration = std::numeric_limits<U64>::max();
    Result resolvedResult = Result::INCOMPLETE;
    U64 generation = 0;
    std::future<Result> installation;
};

PythonDependencyRegistry& DependencyRegistry() {
    // An installation may still be finishing at shutdown. Its future joins
    // before the registry is destroyed, while the operation mutex is still alive.
    (void)PythonOperationMutex();
    static PythonDependencyRegistry registry;
    return registry;
}

void InvalidateDependencyResolution(PythonDependencyRegistry& registry) {
    registry.resolvedGeneration = std::numeric_limits<U64>::max();
    if (registry.request.state != PythonDependencyRequestState::Installing) {
        registry.request = {};
    }
}

}  // namespace

Result StagePythonDependencies(PythonRuntimeContext* context,
                               const std::vector<std::string>& requirements) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot stage dependencies for a null "
                  "context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto [entry, inserted] = registry.records.try_emplace(context);
    if (inserted || entry->second.requirements != requirements) {
        entry->second.requirements = requirements;
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

Result SetPythonDependencyPolicy(const std::string& value) {
    PythonDependencyPolicy policy;
    JST_CHECK(ParsePythonDependencyPolicy(value, policy));

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.dependencyPolicy == policy) {
        return Result::SUCCESS;
    }

    registry.dependencyPolicy = policy;
    InvalidateDependencyResolution(registry);
    ++registry.generation;
    return Result::SUCCESS;
}

Result SetPythonDependencyOrigin(PythonRuntimeContext* context,
                                 const std::string& block,
                                 const std::shared_ptr<Flowgraph::View>& view) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot set the origin for a null "
                  "dependency context.");
        return Result::ERROR;
    }

    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto& record = registry.records[context];
    record.block = block;
    record.view = view;
    return Result::SUCCESS;
}

Result SchedulePythonDependencies(PythonRuntimeContext* context) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot schedule a null dependency "
                  "context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto& record = registry.records[context];
    if (!record.scheduled) {
        record.scheduled = true;
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

Result UnschedulePythonDependencies(PythonRuntimeContext* context) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot unschedule a null dependency "
                  "context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto entry = registry.records.find(context);
    if (entry != registry.records.end() && entry->second.scheduled) {
        entry->second.scheduled = false;
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

Result RemovePythonDependencies(PythonRuntimeContext* context) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot remove a null dependency context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.records.erase(context) > 0) {
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

PythonDependencyStateSnapshot SnapshotPythonDependencyState() {
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    return {
        .generation = registry.generation,
        .request = registry.request,
    };
}

Result ReconcilePythonDependencies() {
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    U64 generation;
    std::vector<std::string> requirements;
    std::vector<PythonDependencyRequestEntry> entries;
    std::optional<PythonDependencyPolicy> dependencyPolicy;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.request.state == PythonDependencyRequestState::Installing) {
            return Result::INCOMPLETE;
        }
        if (registry.resolvedGeneration == registry.generation) {
            return registry.resolvedResult;
        }

        generation = registry.generation;
        dependencyPolicy = registry.dependencyPolicy;
        for (const auto& [_, record] : registry.records) {
            if (!record.scheduled) {
                continue;
            }
            requirements.insert(requirements.end(),
                                record.requirements.begin(),
                                record.requirements.end());
            for (const auto& requirement : record.requirements) {
                entries.push_back({requirement, record.block, record.view});
            }
        }
    }
    std::ranges::sort(requirements);
    requirements.erase(
        std::unique(requirements.begin(), requirements.end()),
        requirements.end());
    std::ranges::sort(entries, [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.requirement, lhs.block) <
               std::tie(rhs.requirement, rhs.block);
    });

    if (requirements.empty()) {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (!registry.hasResolvedDependencies) {
            registry.request = {};
            registry.resolvedGeneration = generation;
            registry.resolvedResult = Result::SUCCESS;
            return Result::SUCCESS;
        }
    }

    PythonDependencyEnvironment environment;
    auto result = PreparePythonDependencyEnvironment(
        requirements, false, environment);
    if (result == Result::INCOMPLETE) {
        const auto policy = dependencyPolicy.has_value()
                                ? *dependencyPolicy
                                : ConfiguredPythonDependencyPolicy();
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            registry.request = {
                .generation = generation,
                .state = policy == PythonDependencyPolicy::Deny
                             ? PythonDependencyRequestState::Denied
                             : PythonDependencyRequestState::ApprovalRequired,
                .dependencies = std::move(entries),
                .message = policy == PythonDependencyPolicy::Deny
                               ? "Dependency installation is disabled by policy."
                               : "",
            };
            registry.resolvedGeneration = generation;
            registry.resolvedResult = Result::INCOMPLETE;
        }
        if (policy == PythonDependencyPolicy::Allow) {
            JST_CHECK(BeginPythonDependencyInstallation(generation));
        }
        return Result::INCOMPLETE;
    }

    if (result == Result::SUCCESS) {
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (registry.generation != generation) {
                return Result::INCOMPLETE;
            }
        }
        result = ActivatePythonDependencyEnvironment(environment);
    }

    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.generation != generation) {
        return Result::INCOMPLETE;
    }
    registry.resolvedGeneration = generation;
    registry.resolvedResult = result == Result::SUCCESS ? Result::SUCCESS
                                                        : Result::INCOMPLETE;
    if (result == Result::SUCCESS) {
        registry.hasResolvedDependencies = !requirements.empty();
        registry.request = {};
    } else if (result == Result::INCOMPLETE) {
        registry.request = {};
    } else {
        registry.request = {
            .generation = generation,
            .state = PythonDependencyRequestState::Failed,
            .dependencies = std::move(entries),
            .message = JST_LOG_LAST_ERROR(),
        };
    }
    return registry.resolvedResult;
}

Result BeginPythonDependencyInstallation(U64 generation) {
    // Destroy the previous future only after releasing the registry mutex.
    std::future<Result> previous;
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.generation != generation ||
        registry.request.generation != generation ||
        (registry.request.state != PythonDependencyRequestState::ApprovalRequired &&
         registry.request.state != PythonDependencyRequestState::Failed)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The dependency approval request is stale.");
        return Result::ERROR;
    }

    const auto policy = registry.dependencyPolicy.has_value()
                            ? *registry.dependencyPolicy
                            : ConfiguredPythonDependencyPolicy();
    if (policy == PythonDependencyPolicy::Deny) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Dependency installation is disabled "
                  "by policy.");
        return Result::ERROR;
    }

    registry.request.state = PythonDependencyRequestState::Installing;
    registry.request.output.clear();
    registry.request.message.clear();
    registry.request.approved = true;
    previous = std::move(registry.installation);
    try {
        registry.installation = std::async(std::launch::async, [generation, &registry]() {
            try {
                return InstallPythonDependencies(generation);
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(registry.mutex);
                registry.request.state = PythonDependencyRequestState::Failed;
                registry.request.message = error.what();
                registry.resolvedGeneration = generation;
                registry.resolvedResult = Result::INCOMPLETE;
                return Result::ERROR;
            }
        });
    } catch (const std::exception& error) {
        registry.request.state = PythonDependencyRequestState::Failed;
        registry.request.message = error.what();
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

namespace {

Result InstallPythonDependencies(U64 generation) {
    auto& registry = DependencyRegistry();
    std::vector<std::string> requirements;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.generation != generation ||
            registry.request.generation != generation ||
            registry.request.state != PythonDependencyRequestState::Installing) {
            if (registry.request.generation == generation &&
                registry.request.state == PythonDependencyRequestState::Installing) {
                registry.request.state = PythonDependencyRequestState::Failed;
                registry.request.message = "The dependency list changed before installation started.";
            }
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The dependency installation request "
                      "is stale.");
            return Result::ERROR;
        }
        for (const auto& dependency : registry.request.dependencies) {
            requirements.push_back(dependency.requirement);
        }
    }
    std::ranges::sort(requirements);
    requirements.erase(std::unique(requirements.begin(), requirements.end()),
                       requirements.end());

    PythonDependencyEnvironment environment;
    const auto onOutput = [&registry, generation](std::string_view output) {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.request.generation != generation ||
            registry.request.state != PythonDependencyRequestState::Installing) {
            return;
        }
        registry.request.output.append(output.data(), output.size());
    };
    const auto prepareResult = PreparePythonDependencyEnvironment(
        requirements, true, environment, onOutput);
    // Serialize the generation check with staging, teardown and activation.
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.generation != generation) {
            registry.request.state = PythonDependencyRequestState::Failed;
            registry.request.message = "The dependency list changed.";
            registry.resolvedGeneration = generation;
            registry.resolvedResult = Result::INCOMPLETE;
            return Result::ERROR;
        }
    }

    auto result = prepareResult;
    if (result == Result::SUCCESS) {
        result = ActivatePythonDependencyEnvironment(environment);
    }

    std::lock_guard<std::mutex> lock(registry.mutex);
    if (result == Result::SUCCESS && registry.generation == generation) {
        registry.request.state = PythonDependencyRequestState::Installed;
        registry.request.message.clear();
        if (!registry.request.output.empty() &&
            !registry.request.output.ends_with('\n')) {
            registry.request.output.push_back('\n');
        }
        registry.request.output += "Installation completed.\n";
        registry.hasResolvedDependencies = !requirements.empty();
        registry.resolvedGeneration = generation;
        registry.resolvedResult = Result::SUCCESS;
        return Result::SUCCESS;
    }

    registry.request.state = PythonDependencyRequestState::Failed;
    registry.request.message = JST_LOG_LAST_ERROR();
    registry.resolvedGeneration = generation;
    registry.resolvedResult = Result::INCOMPLETE;
    return result == Result::SUCCESS ? Result::ERROR : result;
}

}  // namespace

Result ActivatePythonDependencyEnvironment(const PythonDependencyEnvironment& environment) {
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();

    std::vector<PythonRuntimeContext*> contexts;
    PythonDependencyEnvironment previous;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.activeEnvironment.key == environment.key &&
            registry.activeEnvironment.sitePackagesPath ==
                environment.sitePackagesPath) {
            return Result::SUCCESS;
        }

        previous = registry.activeEnvironment;
        for (const auto& [context, record] : registry.records) {
            if (record.scheduled) {
                contexts.push_back(context);
            }
        }
    }
    std::ranges::sort(contexts, std::less<PythonRuntimeContext*>());

    std::vector<PythonRuntimeContext*> unloaded;
    for (auto* context : contexts) {
        const auto result = context->unloadCompute();
        if (result != Result::SUCCESS) {
            for (auto* previousContext : unloaded) {
                (void)previousContext->loadCompute();
            }
            return result;
        }
        unloaded.push_back(context);
    }

    const auto switchResult = SwitchPythonEnvironmentPath(
        previous.sitePackagesPath, environment.sitePackagesPath);
    if (switchResult != Result::SUCCESS) {
        (void)SwitchPythonEnvironmentPath(environment.sitePackagesPath,
                                          previous.sitePackagesPath);
        for (auto* context : contexts) {
            (void)context->loadCompute();
        }
        return switchResult;
    }

    for (auto* context : contexts) {
        const auto result = context->loadCompute();
        if (result == Result::SUCCESS) {
            continue;
        }

        for (auto* loadedContext : contexts) {
            (void)loadedContext->unloadCompute();
        }
        (void)SwitchPythonEnvironmentPath(environment.sitePackagesPath,
                                          previous.sitePackagesPath);
        for (auto* previousContext : contexts) {
            (void)previousContext->loadCompute();
        }
        return result;
    }

    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.activeEnvironment = environment;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream
