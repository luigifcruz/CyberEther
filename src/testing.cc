#include "jetstream/testing.hh"
#include "jetstream/logger.hh"
#include "jetstream/registry.hh"
#include "jetstream/module.hh"
#include "jetstream/runtime.hh"

#include <unordered_map>
#include <unordered_set>

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
    JST_CHECK(Registry::BuildModule(
        pimpl->moduleType,
        pimpl->deviceType,
        pimpl->runtimeType,
        pimpl->providerType,
        pimpl->module));

    TensorMap deviceInputs;
    for (auto& [name, cpuTensor] : pimpl->cpuInputs) {
        if (pimpl->deviceType == DeviceType::CPU) {
            deviceInputs[name] = {"test", name, cpuTensor};
        } else {
            Tensor deviceTensor(pimpl->deviceType, cpuTensor);
            deviceInputs[name] = {"test", name, deviceTensor};
        }
    }

    auto createResult = pimpl->module->create("test", pimpl->config, deviceInputs);
    if (createResult != Result::SUCCESS) {
        JST_ERROR("[TESTING] Failed to create module: {}", pimpl->moduleType);
        return createResult;
    }

    pimpl->runtime = std::make_unique<Runtime>("test", pimpl->deviceType, pimpl->runtimeType);
    auto runtimeCreateResult = pimpl->runtime->create({{"test", pimpl->module}});
    if (runtimeCreateResult != Result::SUCCESS) {
        JST_ERROR("[TESTING] Failed to create runtime: {}", pimpl->moduleType);
        return runtimeCreateResult;
    }

    std::unordered_set<std::string> skippedModules;
    auto computeResult = pimpl->runtime->compute({}, skippedModules);
    if (computeResult != Result::SUCCESS) {
        JST_ERROR("[TESTING] Failed to run compute: {}", pimpl->moduleType);
        return computeResult;
    }

    for (const auto& [name, entry] : pimpl->module->outputs()) {
        if (entry.tensor.device() == DeviceType::CPU) {
            pimpl->cpuOutputs[name] = entry.tensor;
        } else {
            pimpl->cpuOutputs[name] = Tensor(DeviceType::CPU, entry.tensor);
        }
    }

    return Result::SUCCESS;
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
