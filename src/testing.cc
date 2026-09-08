#include "jetstream/testing.hh"
#include "jetstream/logger.hh"
#include "jetstream/registry.hh"
#include "jetstream/module.hh"
#include "jetstream/runtime.hh"

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Jetstream {

struct TestContext::Impl {
    std::string moduleType;
    DeviceType deviceType;
    RuntimeType runtimeType;
    ProviderType providerType;

    std::unordered_map<std::string, Tensor> cpuInputs;
    std::unordered_map<std::string, Tensor> cpuOutputs;
    Parser::Map config;

    std::shared_ptr<Module> module;
    std::unique_ptr<Runtime> runtime;

    ~Impl() {
        (void)cleanup();
    }

    Result cleanup() {
        Result result = Result::SUCCESS;

        if (runtime) {
            const auto destroyResult = runtime->destroy();
            if (destroyResult != Result::SUCCESS && destroyResult != Result::RELOAD) {
                result = destroyResult;
            }
            runtime.reset();
        }

        if (module) {
            const auto destroyResult = module->destroy();
            if (destroyResult != Result::SUCCESS && destroyResult != Result::RELOAD &&
                (result == Result::SUCCESS || result == Result::RELOAD)) {
                result = destroyResult;
            }
            module.reset();
        }

        return result;
    }

    Result snapshotOutputs() {
        for (const auto& [name, entry] : module->outputs()) {
            if (entry.tensor.device() == DeviceType::CPU) {
                cpuOutputs[name] = entry.tensor;
            } else if (entry.tensor.device() == DeviceType::CUDA &&
                       entry.tensor.contiguous() &&
                       entry.tensor.offset() == 0 &&
                       entry.tensor.sizeBytes() == entry.tensor.buffer().sizeBytes()) {
                Tensor cpuOutput;
                JST_CHECK(cpuOutput.create(DeviceType::CPU,
                                           entry.tensor.dtype(),
                                           entry.tensor.shape()));

                Tensor deviceMappedOutput;
                JST_CHECK(deviceMappedOutput.create(entry.tensor.device(), cpuOutput));
                JST_CHECK(deviceMappedOutput.copyFrom(entry.tensor));
                JST_CHECK(cpuOutput.propagateAttributes(entry.tensor));

                cpuOutputs[name] = cpuOutput;
            } else {
                cpuOutputs[name] = Tensor(DeviceType::CPU, entry.tensor);
            }
        }

        return Result::SUCCESS;
    }
};

TestContext::TestContext(const std::string& moduleType,
                         DeviceType device,
                         RuntimeType runtime,
                         const ProviderType& provider)
    : pimpl(std::make_unique<Impl>()) {
    pimpl->moduleType = moduleType;
    pimpl->deviceType = device;
    pimpl->runtimeType = runtime;
    pimpl->providerType = provider;
}

TestContext::~TestContext() = default;

TestContext::TestContext(TestContext&&) noexcept = default;
TestContext& TestContext::operator=(TestContext&&) noexcept = default;

void TestContext::setInput(const std::string& name, Tensor& tensor) {
    pimpl->cpuInputs[name] = tensor;
}

void TestContext::setConfig(const Module::Config& config) {
    config.serialize(pimpl->config);
}

Result TestContext::run() {
    (void)pimpl->cleanup();

    auto result = start();
    if (result == Result::SUCCESS) {
        result = compute();
    }

    (void)pimpl->cleanup();
    return result;
}

Result TestContext::start() {
    if (pimpl->module || pimpl->runtime) {
        JST_ERROR("[TESTING] Test context session is already active: {}", pimpl->moduleType);
        return Result::ERROR;
    }

    JST_CHECK(Registry::BuildModule(
        pimpl->moduleType,
        pimpl->deviceType,
        pimpl->runtimeType,
        pimpl->providerType,
        pimpl->module));

    TensorMap deviceInputs;
    for (auto& [name, cpuTensor] : pimpl->cpuInputs) {
        if (pimpl->deviceType == DeviceType::CPU) {
            deviceInputs[name].requested("test", name);
            deviceInputs[name].tensor = cpuTensor;
        } else {
            Tensor deviceTensor(pimpl->deviceType, cpuTensor);
            deviceInputs[name].requested("test", name);
            deviceInputs[name].tensor = deviceTensor;
        }
    }

    auto createResult = pimpl->module->create("test", pimpl->config, deviceInputs);
    if (createResult != Result::SUCCESS) {
        JST_ERROR("[TESTING] Failed to create module: {}", pimpl->moduleType);
        if (createResult == Result::ERROR) {
            pimpl->module.reset();
        } else {
            pimpl->cleanup();
        }
        return createResult;
    }

    pimpl->runtime = std::make_unique<Runtime>("test", pimpl->deviceType, pimpl->runtimeType);
    auto runtimeCreateResult = pimpl->runtime->create({{"test", pimpl->module}});
    if (runtimeCreateResult != Result::SUCCESS) {
        JST_ERROR("[TESTING] Failed to create runtime: {}", pimpl->moduleType);
        pimpl->cleanup();
        return runtimeCreateResult;
    }

    return Result::SUCCESS;
}

Result TestContext::compute() {
    if (!pimpl->module || !pimpl->runtime) {
        JST_ERROR("[TESTING] Test context session is not active: {}", pimpl->moduleType);
        return Result::ERROR;
    }

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    const auto computeResult = pimpl->runtime->compute({}, skippedModules, failedModules);
    if (computeResult != Result::SUCCESS) {
        JST_ERROR("[TESTING] Failed to run compute: {}", pimpl->moduleType);
        return computeResult;
    }

    return pimpl->snapshotOutputs();
}

Result TestContext::reconfigure(const Module::Config& config, const bool validateOnly) {
    if (!pimpl->module || !pimpl->runtime) {
        JST_ERROR("[TESTING] Test context session is not active: {}", pimpl->moduleType);
        return Result::ERROR;
    }

    Parser::Map candidate;
    JST_CHECK(config.serialize(candidate));
    const auto result = pimpl->module->reconfigure(candidate, validateOnly);
    if (!validateOnly && (result == Result::SUCCESS ||
                          result == Result::RELOAD ||
                          result == Result::RECREATE)) {
        pimpl->config = std::move(candidate);
    }
    return result;
}

Result TestContext::stop() {
    return pimpl->cleanup();
}

Tensor& TestContext::output(const std::string& name) {
    auto it = pimpl->cpuOutputs.find(name);
    if (it == pimpl->cpuOutputs.end()) {
        JST_FATAL("[TESTING] Output not found: {}", name);
        JST_CHECK_THROW(Result::ERROR);
    }
    return it->second;
}

DeviceType TestContext::device() const {
    return pimpl->deviceType;
}

RuntimeType TestContext::runtime() const {
    return pimpl->runtimeType;
}

const ProviderType& TestContext::provider() const {
    return pimpl->providerType;
}

}  // namespace Jetstream
