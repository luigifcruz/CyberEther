#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <thread>
#include <unordered_set>
#include <utility>

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
    explicit PythonRuntimeSmokeModule(std::string source = R"PY(
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
                                      std::unordered_map<std::string, std::string> pieces = {})
        : source(std::move(source)), pieces(std::move(pieces)) {}

    Result define() override {
        JST_CHECK(defineInterfaceInput("in"));
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        input = inputs().at("in").tensor;
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, input.shape()));
        outputs()["out"].produced(name(), "out", output);

        return createCompute(source,
                             pieces,
                             {"in"},
                             inputs(),
                             {"out"},
                             outputs());
    }

    std::string source;
    std::unordered_map<std::string, std::string> pieces;
    Tensor input;
    Tensor output;
};

std::shared_ptr<Module> makePythonRuntimeSmokeModule(std::string source = {},
                                                     std::unordered_map<std::string, std::string> pieces = {}) {
    auto impl = source.empty() && pieces.empty()
                    ? std::make_shared<PythonRuntimeSmokeModule>()
                    : std::make_shared<PythonRuntimeSmokeModule>(std::move(source), std::move(pieces));
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

void destroyPythonCompute(const std::shared_ptr<Module>& module) {
    auto pythonContext = std::dynamic_pointer_cast<PythonRuntimeContext>(module->context()->runtime());
    REQUIRE(pythonContext != nullptr);
    REQUIRE(pythonContext->destroyCompute() == Result::SUCCESS);
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
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime expands source pieces before compiling compute()", "[runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
def compute(ctx):
    <<<BODY>>>
)PY", {{"BODY", "ctx.outputs[0][...] = ctx.inputs[0] + 5.0"}});

    Parser::Map config;
    const auto createResult = module->create("python_runtime_piece_smoke", config, inputs);

    if (createResult != Result::SUCCESS && optionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime source-piece test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_piece_smoke", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_piece_smoke"}, skippedModules, failedModules) == Result::SUCCESS);

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>(i + 6)) < 1e-5f);
    }

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
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
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime handles NumPy buffering on a worker thread", "[runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {16}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
import gc as _jetstream_gc
import numpy as _jetstream_np

_state = globals().setdefault('_numpy_buffer_state', {
    'buffer': _jetstream_np.zeros(0, dtype=_jetstream_np.float32),
    'last': 0.0,
})

def compute(ctx):
    mono = _jetstream_np.asarray(ctx.inputs[0], dtype=_jetstream_np.float32).reshape(-1)
    if mono.size == 0:
        return

    for _ in range(16):
        owned = mono.copy()
        decimated = owned[::2]
        _state['buffer'] = _jetstream_np.concatenate([_state['buffer'], decimated])

        if _state['buffer'].size >= 8:
            chunk = _state['buffer'][:8]
            _state['buffer'] = _state['buffer'][8:]
            peak = float(_jetstream_np.max(_jetstream_np.abs(chunk))) if chunk.size else 0.0
            if peak > 1e-6:
                chunk = chunk / max(1.0, peak)
            _state['last'] = float(_jetstream_np.sum(chunk))

    ctx.outputs[0][...] = ctx.inputs[0] * 7.0
    _jetstream_gc.collect()
)PY");
    Parser::Map config;
    const auto createResult = module->create("python_runtime_numpy_buffer", config, inputs);

    if (createResult != Result::SUCCESS && optionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime NumPy test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    auto runtime = std::make_shared<Runtime>("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime->create({{"python_runtime_numpy_buffer", module}}) == Result::SUCCESS);

    auto done = std::make_shared<std::atomic<bool>>(false);
    auto computeResult = std::make_shared<std::atomic<Result>>(Result::ERROR);

    std::thread computeThread([runtime, done, computeResult] {
        for (int iteration = 0; iteration < 16; ++iteration) {
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            const auto result = runtime->compute({"python_runtime_numpy_buffer"},
                                                 skippedModules,
                                                 failedModules);
            if (result != Result::SUCCESS || !skippedModules.empty() || !failedModules.empty()) {
                computeResult->store(result == Result::SUCCESS ? Result::ERROR : result,
                                     std::memory_order_release);
                done->store(true, std::memory_order_release);
                return;
            }
        }

        computeResult->store(Result::SUCCESS, std::memory_order_release);
        done->store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!done->load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!done->load(std::memory_order_acquire)) {
        computeThread.detach();
        FAIL("Python runtime NumPy buffering blocked on a worker thread.");
    }

    computeThread.join();
    REQUIRE(computeResult->load(std::memory_order_acquire) == Result::SUCCESS);

    Tensor output = module->outputs().at("out").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>((i + 1) * 7)) < 1e-5f);
    }

    REQUIRE(runtime->destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime runs cleanup hook during compute destruction", "[runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0] * 11.0

def cleanup():
    print("__JETSTREAM_CLEANUP_CALLED__")
)PY");
    Parser::Map config;
    const auto createResult = module->create("python_runtime_cleanup", config, inputs);

    if (createResult != Result::SUCCESS && optionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime cleanup test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_cleanup", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_cleanup"}, skippedModules, failedModules) == Result::SUCCESS);

    auto pythonContext = std::dynamic_pointer_cast<PythonRuntimeContext>(module->context()->runtime());
    REQUIRE(pythonContext != nullptr);
    REQUIRE(pythonContext->destroyCompute() == Result::SUCCESS);

    const auto diagnostic = pythonContext->diagnostic();
    bool cleanupCalled = false;
    for (const auto& line : diagnostic.console) {
        cleanupCalled = cleanupCalled || line.find("__JETSTREAM_CLEANUP_CALLED__") != std::string::npos;
    }
    REQUIRE(cleanupCalled);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python runtime cleans multiprocessing resources on compute destruction", "[runtime][python]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["in"].produced("source", "out", input);

    auto module = makePythonRuntimeSmokeModule(R"PY(
try:
    import multiprocessing as mp
    mp_semaphore = mp.get_context("spawn").Semaphore(1)
except Exception:
    mp_semaphore = None

def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0] * 13.0
)PY");
    Parser::Map config;
    const auto createResult = module->create("python_runtime_multiprocessing_cleanup", config, inputs);

    if (createResult != Result::SUCCESS && optionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime multiprocessing cleanup test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_multiprocessing_cleanup", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_multiprocessing_cleanup"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    destroyPythonCompute(module);
    REQUIRE(module->destroy() == Result::SUCCESS);

    auto verifier = makePythonRuntimeSmokeModule(R"PY(
def compute(ctx):
    import sys as _jetstream_sys
    import multiprocessing.util as _jetstream_mp_util

    pending = [
        key for key in _jetstream_mp_util._finalizer_registry
        if key[0] is not None and key[0] >= 0
    ]

    tracker = _jetstream_sys.modules.get("multiprocessing.resource_tracker")
    tracker_fd = -1 if tracker is None or tracker._resource_tracker._fd is None else tracker._resource_tracker._fd

    ctx.outputs[0][...] = 0.0
    ctx.outputs[0][0] = len(pending)
    ctx.outputs[0][1] = 0.0 if tracker_fd < 0 else 1.0
)PY");
    const auto verifyCreateResult = verifier->create("python_runtime_multiprocessing_verify", config, inputs);

    if (verifyCreateResult != Result::SUCCESS && optionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime multiprocessing verify test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(verifyCreateResult == Result::SUCCESS);

    Runtime verifyRuntime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(verifyRuntime.create({{"python_runtime_multiprocessing_verify", verifier}}) == Result::SUCCESS);

    skippedModules.clear();
    failedModules.clear();
    const auto verifyComputeResult = verifyRuntime.compute({"python_runtime_multiprocessing_verify"},
                                                           skippedModules,
                                                           failedModules);
    const auto verifyDiagnostic = verifier->context()->runtime()->diagnostic();
    for (const auto& line : verifyDiagnostic.console) {
        INFO(line);
    }
    REQUIRE(verifyComputeResult == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());

    Tensor verifyOutput = verifier->outputs().at("out").tensor;
    const auto* verifyData = verifyOutput.data<F32>();
    REQUIRE(verifyData != nullptr);
    REQUIRE(verifyData[0] == 0.0f);
    REQUIRE(verifyData[1] == 0.0f);

    REQUIRE(verifyRuntime.destroy() == Result::SUCCESS);
    destroyPythonCompute(verifier);
    REQUIRE(verifier->destroy() == Result::SUCCESS);
}
