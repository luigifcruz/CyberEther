#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <jetstream/memory/axis.hh>

#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/platform.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_python.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"
#include "runtime/python/dependencies/coordinator.hh"

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
    Result loadCompute() final;

 private:
    Result loadComputeSource(const std::string& source);

    std::string computeSource;
    std::string computeFile;
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

Result PythonImplPython::loadComputeSource(const std::string& source) {
    JST_CHECK(SetPythonDependencyOrigin(this, name(), view()));
    const auto computeResult = createCompute(source,
                                             {},
                                             inputPortOrder(),
                                             inputs(),
                                             outputPortOrder(),
                                             outputs(),
                                             environment(),
                                             view(),
                                             computeFile);
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

Result PythonImplPython::loadCompute() {
    return loadComputeSource(computeSource);
}

Result PythonImplPython::create() {
    JST_CHECK(PythonImpl::create());

    computeSource = code;
    computeFile.clear();
    if (source == "file") {
        if (file.empty()) {
            JST_ERROR("[PYTHON] Choose a Python file to run.");
            return Result::INCOMPLETE;
        }

        const auto path = Platform::PathFromUtf8(file);
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error)) {
            JST_ERROR("[PYTHON] Python file '{}' is not a readable regular file.", file);
            return Result::INCOMPLETE;
        }
        std::ifstream stream(path, std::ios::binary);
        if (!stream.is_open()) {
            JST_ERROR("[PYTHON] Can't open Python file '{}'.", file);
            return Result::INCOMPLETE;
        }
        std::ostringstream contents;
        contents << stream.rdbuf();
        if (stream.bad() || contents.bad()) {
            JST_ERROR("[PYTHON] Can't read Python file '{}'.", file);
            return Result::INCOMPLETE;
        }
        computeSource = contents.str();
        const auto absolutePath = std::filesystem::absolute(path, error);
        computeFile = Platform::PathToUtf8(error ? path : absolutePath);
    }

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

    JST_CHECK(loadCompute());

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

    if (config.source != source ||
        (config.source == "file" && config.file != file) ||
        config.inputCount != inputCount ||
        config.outputCount != outputCount ||
        config.outputTensorSpecs != outputTensorSpecs ||
        config.throttled != throttled) {
        return Result::RECREATE;
    }

    if (config.source == "editor" && config.code != code) {
        JST_CHECK(loadComputeSource(config.code));
        computeSource = config.code;
    }
    code = config.code;
    file = config.file;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PythonImplPython, DeviceType::CPU, RuntimeType::PYTHON, "generic");

}  // namespace Jetstream::Modules
