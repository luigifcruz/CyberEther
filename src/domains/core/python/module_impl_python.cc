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
    Result create() final;
    Result destroy() final;
    Result reconfigure() final;

 private:
    Result loadCompute(const std::string& source);
};

Result PythonImplPython::loadCompute(const std::string& source) {
    const auto computeResult = createCompute(source,
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
        config.outputTensorSpecs != outputTensorSpecs) {
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
