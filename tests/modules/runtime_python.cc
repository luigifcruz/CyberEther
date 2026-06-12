#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>

#include "jetstream/detail/module_impl.hh"
#include "jetstream/logger.hh"
#include "jetstream/module_context.hh"
#include "jetstream/runtime_context_python.hh"
#include "jetstream/scheduler_context.hh"

namespace {

using namespace Jetstream;

struct PythonRuntimeSmokeConfig : Module::Config {
    JST_MODULE_TYPE(python_runtime_smoke)

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct PythonRuntimeSmokeModule : Module::Impl,
                                  DynamicConfig<PythonRuntimeSmokeConfig>,
                                  PythonRuntimeContext,
                                  Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceInput("in"));
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        input = inputs().at("in").tensor;
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, input.shape()));
        outputs()["out"].produced(name(), "out", output);

        return createCompute(R"PY(
_seen = None

def compute(ctx):
    global _seen
    current = (id(ctx), id(ctx.inputs[0]), id(ctx.outputs[0]))
    if _seen is None:
        _seen = current
    elif current != _seen:
        raise RuntimeError("Python context arrays changed")
    ctx.outputs[0][...] = ctx.inputs[0] * 3.0
)PY",
                             {"in"},
                             inputs(),
                             {"out"},
                             outputs());
    }

    Tensor input;
    Tensor output;
};

std::shared_ptr<Module> makePythonRuntimeSmokeModule() {
    auto impl = std::make_shared<PythonRuntimeSmokeModule>();
    auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
    auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
    auto context = std::make_shared<Module::Context>(runtimeContext,
                                                     schedulerContext,
                                                     nullptr,
                                                     nullptr);
    auto stagedConfig = std::static_pointer_cast<Module::Config>(impl);
    auto candidateConfig = std::static_pointer_cast<Module::Config>(impl->candidate());

    return std::make_shared<Module>(DeviceType::CPU,
                                    RuntimeType::PYTHON,
                                    "generic",
                                    impl,
                                    context,
                                    stagedConfig,
                                    candidateConfig);
}

bool optionalPythonRuntimeUnavailable() {
    const auto& error = JST_LOG_LAST_ERROR();
    return error.find("Can't load Python library") != std::string::npos ||
           error.find("Can't initialize Python runtime helpers") != std::string::npos ||
           error.find("Can't load Python symbol") != std::string::npos ||
           error.find("Auto could not find a valid Python runtime") != std::string::npos ||
           error.find("No libpython was found") != std::string::npos ||
           error.find("No loadable libpython was found") != std::string::npos;
}

}  // namespace

TEST_CASE("Python runtime executes compute() with tensor inputs and outputs", "[runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule();
    Parser::Map config;
    const auto createResult = module->create("python_runtime_smoke", config, inputs);

    if (createResult != Result::SUCCESS && optionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime smoke test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_smoke", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_smoke"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(runtime.compute({"python_runtime_smoke"}, skippedModules, failedModules) == Result::SUCCESS);

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>((i + 1) * 3)) < 1e-5f);
    }

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime compute can run on a different thread than creation", "[runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule();
    Parser::Map config;
    const auto createResult = module->create("python_runtime_smoke", config, inputs);

    if (createResult != Result::SUCCESS && optionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime thread test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    auto runtime = std::make_shared<Runtime>("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime->create({{"python_runtime_smoke", module}}) == Result::SUCCESS);

    auto done = std::make_shared<std::atomic<bool>>(false);
    auto computeResult = std::make_shared<std::atomic<Result>>(Result::ERROR);

    std::thread computeThread([runtime, done, computeResult] {
        std::unordered_set<std::string> skippedModules;
        std::unordered_set<std::string> failedModules;
        computeResult->store(runtime->compute({"python_runtime_smoke"}, skippedModules, failedModules),
                             std::memory_order_release);
        done->store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!done->load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!done->load(std::memory_order_acquire)) {
        computeThread.detach();
        FAIL("Python runtime compute blocked on a worker thread.");
    }

    computeThread.join();
    REQUIRE(computeResult->load(std::memory_order_acquire) == Result::SUCCESS);

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>((i + 1) * 3)) < 1e-5f);
    }

    REQUIRE(runtime->destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}
