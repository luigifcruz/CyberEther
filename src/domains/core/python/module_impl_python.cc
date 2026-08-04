#include <array>
#include <limits>
#include <unordered_set>
#include <vector>

#include <jetstream/memory/axis.hh>

#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_python.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

bool PythonRuntimeUnavailable(const PythonRuntimeContext::Diagnostic& diagnostic) {
    for (const auto& line : diagnostic.console) {
        if (line.find("Can't load Python library") != std::string::npos ||
            line.find("Can't initialize Python runtime helpers") != std::string::npos ||
            line.find("Can't load Python symbol") != std::string::npos ||
            line.find("Auto could not find a valid Python runtime") != std::string::npos ||
            line.find("No libpython was found") != std::string::npos ||
            line.find("No loadable libpython was found") != std::string::npos) {
            return true;
        }
    }

    return false;
}

}  // namespace

struct PythonImplPython : public PythonImpl,
                          public PythonRuntimeContext,
                          public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;
    Result destroy() final;
    Result reconfigure() final;

 private:
    Result loadCompute(const std::string& source);
};

Result PythonImplPython::validate() {
    const auto previousOutputPlan = candidateOutputPlan;
    JST_CHECK(PythonImpl::validate());

    for (U64 i = 0; i < candidateOutputPlan.size(); ++i) {
        const auto& output = candidateOutputPlan[i];
        const auto label = "output" + std::to_string(i);
        if (output.device != DeviceType::CPU && output.device != DeviceType::CUDA) {
            JST_ERROR("[PYTHON] Python tensor {} device must be CPU or CUDA (got {}).",
                      label, output.device);
            candidateOutputPlan = previousOutputPlan;
            return Result::ERROR;
        }

#ifndef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (output.device == DeviceType::CUDA) {
            JST_ERROR("[PYTHON] Python tensor {} requires the unavailable CUDA backend.",
                      label);
            candidateOutputPlan = previousOutputPlan;
            return Result::ERROR;
        }
#endif

        if (output.device == DeviceType::CPU || output.device == DeviceType::CUDA) {
            U64 alignedSize = 0;
            if (!detail::CheckedPageAlignedSize(output.sizeBytes, alignedSize) ||
                alignedSize > std::numeric_limits<std::size_t>::max()) {
                JST_ERROR("[PYTHON] Python tensor {} allocation size is too large.", label);
                candidateOutputPlan = previousOutputPlan;
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

Result PythonImplPython::loadCompute(const std::string& source) {
    const auto computeResult = createCompute(source,
                                             {},
                                             inputPortOrder(),
                                             inputs(),
                                             outputPortOrder(),
                                             outputs(),
                                             environment(),
                                             view());
    if (computeResult == Result::SUCCESS) {
        return Result::SUCCESS;
    }

    const auto currentDiagnostic = diagnostic();
    if (PythonRuntimeUnavailable(currentDiagnostic)) {
        return computeResult;
    }

    if (currentDiagnostic.status == "Source error.") {
        return Result::SUCCESS;
    }

    return computeResult;
}

Result PythonImplPython::create() {
    JST_CHECK(PythonImpl::create());

    const std::array<std::string_view, 3> axisAttributes = {
        SampleAxisAttribute,
        BatchAxisAttribute,
        ChannelAxisAttribute,
    };

    std::vector<std::unordered_set<std::string>> immutableKeys;
    immutableKeys.reserve(candidateOutputPlan.size());
    for (const auto& plan : candidateOutputPlan) {
        std::unordered_set<std::string> keys;
        if (!plan.attributes.empty()) {
            for (const auto& name : axisAttributes) {
                keys.insert(std::string(name));
            }
        }
        immutableKeys.push_back(std::move(keys));
    }
    setImmutableOutputAttributes(immutableKeys);

    JST_CHECK(loadCompute(code));

    return Result::SUCCESS;
}

Result PythonImplPython::destroy() {
    JST_CHECK(destroyCompute());
    JST_CHECK(PythonImpl::destroy());

    return Result::SUCCESS;
}

Result PythonImplPython::reconfigure() {
    auto config = *candidate();
    normalizeOutputSpecs(config);

    if (config.inputCount != inputCount ||
        config.outputCount != outputCount ||
        config.outputTensorSpecs != outputTensorSpecs ||
        config.throttled != throttled) {
        return Result::RECREATE;
    }

    if (config.code != code) {
        JST_CHECK(loadCompute(config.code));
        code = config.code;
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PythonImplPython, DeviceType::CPU, RuntimeType::PYTHON, "generic");

}  // namespace Jetstream::Modules
