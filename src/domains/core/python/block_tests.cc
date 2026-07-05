#include <catch2/catch_test_macros.hpp>

#include <any>
#include <cmath>
#include <string>

#include "jetstream/block_interface.hh"
#include "jetstream/detail/block_impl.hh"
#include "jetstream/domains/core/python/block.hh"
#include "jetstream/flowgraph_environment.hh"
#include "jetstream/logger.hh"
#include "jetstream/runtime_context.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

bool OptionalPythonRuntimeUnavailableForBlock() {
    const auto& error = JST_LOG_LAST_ERROR();
    return error.find("Can't load Python library") != std::string::npos ||
           error.find("Can't initialize Python runtime helpers") != std::string::npos ||
           error.find("Can't load Python symbol") != std::string::npos ||
           error.find("Auto could not find a valid Python runtime") != std::string::npos ||
           error.find("No libpython was found") != std::string::npos ||
           error.find("No loadable libpython was found") != std::string::npos;
}

struct PythonMetricsSourceTestConfig : Block::Config {
    JST_BLOCK_TYPE(python_metrics_source_test)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Python Metrics Source Test",
                          "Exposes public metrics for Python block tests.",
                          "A test-only block used by Python block metric coverage.")

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

struct PythonMetricsSourceTestBlock : Block::Impl,
                                      DynamicConfig<PythonMetricsSourceTestConfig> {
    Result define() override {
        JST_CHECK(defineInterfaceMetric("answer",
                                        "Answer",
                                        "Public test metric.",
                                        "label",
            []() -> std::any {
                return I64{42};
            }));
        JST_CHECK(defineInterfaceMetric("secret",
                                        "Secret",
                                        "Private test metric.",
                                        "private-label",
            []() -> std::any {
                return std::string("hidden");
            }));
        return Result::SUCCESS;
    }
};

JST_REGISTER_BLOCK(PythonMetricsSourceTestBlock);

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block exposes dynamic ports from counts",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.inputCount = 2;
    config.outputCount = 3;

    REQUIRE(flowgraph->blockCreate("python_ports", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    const auto block = viewBlock("python_ports");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(block.interfaceInputs.size() == 2);
    REQUIRE(block.interfaceOutputs.size() == 3);
    REQUIRE(block.interfaceConfigs.size() == 7);
    REQUIRE(block.interfaceInputs.at(0).name == "input0");
    REQUIRE(block.interfaceInputs.at(1).name == "input1");
    REQUIRE(block.interfaceOutputs.at(0).name == "output0");
    REQUIRE(block.interfaceOutputs.at(2).name == "output2");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block can create and reconfigure a zero-port compute module",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    pass\n";
    config.inputCount = 0;
    config.outputCount = 0;

    REQUIRE(flowgraph->blockCreate("python_zero", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_zero");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block create test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.inputs.empty());
    REQUIRE(block.outputs.empty());

    Parser::Map nextConfig;
    nextConfig["code"] = std::string("def compute(ctx):\n    pass\n");
    nextConfig["inputCount"] = U64{0};
    nextConfig["outputCount"] = U64{0};

    REQUIRE(flowgraph->blockReconfigure("python_zero", nextConfig) == Result::SUCCESS);
    block = viewBlock("python_zero");
    REQUIRE(block.state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block recreates the module when throttled changes",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    ctx.outputs[0][...] = 1.0\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[4]"}};

    REQUIRE(flowgraph->blockCreate("python_throttle", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_throttle");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block throttle test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    const auto previousTensorId = block.outputs.at("output0").tensor.id();

    Parser::Map nextConfig = block.config;
    nextConfig["throttled"] = true;

    REQUIRE(flowgraph->blockReconfigure("python_throttle", nextConfig) == Result::SUCCESS);

    block = viewBlock("python_throttle");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(std::any_cast<bool>(block.config.at("throttled")) == true);
    REQUIRE(block.outputs.at("output0").tensor.id() != previousTensorId);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block creates a configured output without inputs",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    ctx.outputs[0][...] = 3.0\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[4]"}};

    REQUIRE(flowgraph->blockCreate("python_source", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_source");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block source test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.inputs.empty());
    REQUIRE(block.outputs.size() == 1);

    const Tensor& output = block.outputs.at("output0").tensor;
    REQUIRE(output.dtype() == DataType::F32);
    REQUIRE(output.shape() == Shape{4});

    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_source");
    const Tensor& computed = block.outputs.at("output0").tensor;
    for (Index i = 0; i < computed.size(); ++i) {
        REQUIRE(std::abs(computed.at<F32>(i) - 3.0f) < 1e-5f);
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block reads and writes flowgraph environment",
                 "[modules][python][block]") {
    Parser::Map station;
    station["frequency"] = static_cast<F64>(100.5);
    REQUIRE(flowgraph->environment().set("station", station) == Result::SUCCESS);

    Blocks::Python config;
    config.code =
        "_n = 0\n"
        "\n"
        "def compute(ctx):\n"
        "    global _n\n"
        "    _n += 1\n"
        "    station = ctx.env.get(\"station\", {})\n"
        "    ctx.outputs[0][...] = station.get(\"frequency\", -1.0)\n"
        "    if _n == 1:\n"
        "        ctx.env[\"telemetry\"] = {\"count\": 9007199254740993, \"label\": \"ok\"}\n"
        "    else:\n"
        "        ctx.env[\"telemetry\"][\"count\"] = _n\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[1]"}};

    REQUIRE(flowgraph->blockCreate("python_env", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_env");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block environment test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_env");
    const Tensor& output = block.outputs.at("output0").tensor;
    REQUIRE(std::abs(output.at<F32>(0) - 100.5f) < 1e-3f);

    Parser::Map telemetry;
    REQUIRE(flowgraph->environment().get("telemetry", telemetry) == Result::SUCCESS);
    REQUIRE(telemetry.contains("count"));
    REQUIRE(std::any_cast<I64>(telemetry.at("count")) == 9007199254740993LL);
    REQUIRE(telemetry.contains("label"));
    REQUIRE(std::any_cast<std::string>(telemetry.at("label")) == "ok");

    station["frequency"] = static_cast<F64>(200.25);
    REQUIRE(flowgraph->environment().set("station", station) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_env");
    const Tensor& updated = block.outputs.at("output0").tensor;
    REQUIRE(std::abs(updated.at<F32>(0) - 200.25f) < 1e-3f);

    REQUIRE(flowgraph->environment().get("telemetry", telemetry) == Result::SUCCESS);
    REQUIRE(std::any_cast<I64>(telemetry.at("count")) == 2);
    REQUIRE(std::any_cast<std::string>(telemetry.at("label")) == "ok");

    REQUIRE(flowgraph->environment().clear("station") == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_env");
    const Tensor& cleared = block.outputs.at("output0").tensor;
    REQUIRE(std::abs(cleared.at<F32>(0) - (-1.0f)) < 1e-3f);

    REQUIRE(flowgraph->environment().get("telemetry", telemetry) == Result::SUCCESS);
    REQUIRE(std::any_cast<I64>(telemetry.at("count")) == 3);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block round-trips complex environment values",
                 "[modules][python][block]") {
    Parser::Map carrier;
    carrier["iq"] = CF32{1.5f, -2.5f};
    carrier["reference"] = CF64{3.25, 4.75};
    carrier["taps"] = std::vector<CF32>{{1.0f, 1.0f}, {2.0f, -2.0f}};
    REQUIRE(flowgraph->environment().set("carrier", carrier) == Result::SUCCESS);

    Blocks::Python config;
    config.code =
        "def compute(ctx):\n"
        "    carrier = ctx.env.get(\"carrier\", {})\n"
        "    iq = carrier.get(\"iq\", 0j)\n"
        "    reference = carrier.get(\"reference\", 0j)\n"
        "    taps = carrier.get(\"taps\", ())\n"
        "    ctx.outputs[0][...] = abs(iq)\n"
        "    carrier[\"iq\"] = iq * 2\n"
        "    carrier[\"taps\"] = [tap * 2 for tap in taps]\n"
        "    ctx.env[\"mirror\"] = {\n"
        "        \"value\": complex(iq.real, -iq.imag),\n"
        "        \"reference_magnitude\": abs(reference),\n"
        "        \"spectrum\": [complex(0.0, 1.0), complex(2.0, 3.0)],\n"
        "    }\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[1]"}};

    REQUIRE(flowgraph->blockCreate("python_complex", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_complex");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block complex environment test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_complex");
    const Tensor& output = block.outputs.at("output0").tensor;
    REQUIRE(std::abs(output.at<F32>(0) - std::abs(CF32{1.5f, -2.5f})) < 1e-4f);

    Parser::Map updated;
    REQUIRE(flowgraph->environment().get("carrier", updated) == Result::SUCCESS);
    const auto iq = std::any_cast<CF32>(updated.at("iq"));
    REQUIRE(std::abs(iq.real() - 3.0f) < 1e-5f);
    REQUIRE(std::abs(iq.imag() + 5.0f) < 1e-5f);
    const auto reference = std::any_cast<CF64>(updated.at("reference"));
    REQUIRE(std::abs(reference.real() - 3.25) < 1e-9);
    REQUIRE(std::abs(reference.imag() - 4.75) < 1e-9);
    const auto taps = std::any_cast<std::vector<CF32>>(updated.at("taps"));
    REQUIRE(taps.size() == 2);
    REQUIRE(std::abs(taps[0].real() - 2.0f) < 1e-5f);
    REQUIRE(std::abs(taps[0].imag() - 2.0f) < 1e-5f);
    REQUIRE(std::abs(taps[1].real() - 4.0f) < 1e-5f);
    REQUIRE(std::abs(taps[1].imag() + 4.0f) < 1e-5f);

    Parser::Map mirror;
    REQUIRE(flowgraph->environment().get("mirror", mirror) == Result::SUCCESS);
    const auto value = std::any_cast<CF64>(mirror.at("value"));
    REQUIRE(std::abs(value.real() - 1.5) < 1e-5);
    REQUIRE(std::abs(value.imag() - 2.5) < 1e-5);
    const auto magnitude = std::any_cast<F64>(mirror.at("reference_magnitude"));
    REQUIRE(std::abs(magnitude - std::abs(CF64{3.25, 4.75})) < 1e-9);
    const auto spectrum = std::any_cast<std::vector<CF64>>(mirror.at("spectrum"));
    REQUIRE(spectrum.size() == 2);
    REQUIRE(std::abs(spectrum[0].real()) < 1e-9);
    REQUIRE(std::abs(spectrum[0].imag() - 1.0) < 1e-9);
    REQUIRE(std::abs(spectrum[1].real() - 2.0) < 1e-9);
    REQUIRE(std::abs(spectrum[1].imag() - 3.0) < 1e-9);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block round-trips null environment values",
                 "[modules][python][block]") {
    Parser::Map station;
    station["label"] = std::string("otl");
    station["offset"] = std::any();
    REQUIRE(flowgraph->environment().set("station", station) == Result::SUCCESS);

    Blocks::Python config;
    config.code =
        "def compute(ctx):\n"
        "    station = ctx.env.get(\"station\", {})\n"
        "    ctx.outputs[0][...] = 1.0 if station.get(\"offset\") is None else 0.0\n"
        "    ctx.env[\"status\"] = {\n"
        "        \"value\": None,\n"
        "        \"items\": [None, 1.0],\n"
        "        \"nested\": {\"missing\": None},\n"
        "    }\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[1]"}};

    REQUIRE(flowgraph->blockCreate("python_null", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_null");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block null environment test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_null");
    const Tensor& output = block.outputs.at("output0").tensor;
    REQUIRE(output.at<F32>(0) == 1.0f);

    Parser::Map status;
    REQUIRE(flowgraph->environment().get("status", status) == Result::SUCCESS);
    REQUIRE(!status.at("value").has_value());
    const auto& items = std::any_cast<const Parser::Sequence&>(status.at("items"));
    REQUIRE(items.size() == 2);
    REQUIRE(!items[0].has_value());
    REQUIRE(std::abs(std::any_cast<F64>(items[1]) - 1.0) < 1e-9);
    const auto& nested = std::any_cast<const Parser::Map&>(status.at("nested"));
    REQUIRE(!nested.at("missing").has_value());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block stores numpy values with matching widths",
                 "[modules][python][block]") {
    Parser::Map seeded;
    seeded["level"] = F64{0.0};
    REQUIRE(flowgraph->environment().set("seeded", seeded) == Result::SUCCESS);

    Blocks::Python config;
    config.code =
        "_cycle = [0]\n"
        "\n"
        "def compute(ctx):\n"
        "    try:\n"
        "        import numpy as np\n"
        "    except Exception:\n"
        "        ctx.env[\"numpy_status\"] = {\"available\": False}\n"
        "        return\n"
        "    _cycle[0] += 1\n"
        "    if _cycle[0] == 1:\n"
        "        ctx.env[\"numpy_status\"] = {\"available\": True}\n"
        "        ctx.env[\"sensor\"] = {\n"
        "            \"gain\": np.float32(1.5),\n"
        "            \"count\": np.int16(-7),\n"
        "            \"index\": np.uint32(9),\n"
        "            \"big\": np.uint64(2**63 + 1),\n"
        "            \"flag\": np.bool_(True),\n"
        "            \"off\": np.array(False),\n"
        "            \"iq\": np.complex64(1 + 2j),\n"
        "            \"wide\": np.complex128(3 - 4j),\n"
        "            \"taps\": np.array([1 + 1j, 2 - 2j], dtype=np.complex64),\n"
        "            \"window\": np.array([0.25, 0.75], dtype=np.float32),\n"
        "        }\n"
        "        ctx.env[\"seeded\"][\"level\"] = np.float32(2.5)\n"
        "        return\n"
        "    if _cycle[0] == 2:\n"
        "        sensor = ctx.env.get(\"sensor\", {})\n"
        "        taps = sensor.get(\"taps\")\n"
        "        ctx.env[\"report\"] = {\n"
        "            \"gain_type\": type(sensor.get(\"gain\")).__name__,\n"
        "            \"count_type\": type(sensor.get(\"count\")).__name__,\n"
        "            \"iq_type\": type(sensor.get(\"iq\")).__name__,\n"
        "            \"taps_type\": type(taps).__name__,\n"
        "            \"taps_dtype\": str(getattr(taps, \"dtype\", \"\")),\n"
        "            \"taps_writable\": bool(getattr(taps, \"flags\", None) and taps.flags.writeable),\n"
        "            \"level_type\": type(ctx.env.get(\"seeded\", {}).get(\"level\")).__name__,\n"
        "        }\n"
        "        ctx.env[\"burst\"] = {\n"
        "            \"window\": np.arange(4, dtype=np.float32) * 0.5,\n"
        "            \"taps\": taps * 2,\n"
        "        }\n";
    config.inputCount = 0;
    config.outputCount = 0;

    REQUIRE(flowgraph->blockCreate("python_numpy", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_numpy");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block numpy test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map status;
    REQUIRE(flowgraph->environment().get("numpy_status", status) == Result::SUCCESS);
    if (!std::any_cast<bool>(status.at("available"))) {
        SUCCEED("Skipping Python block numpy test because NumPy is unavailable.");
        return;
    }

    Parser::Map sensor;
    REQUIRE(flowgraph->environment().get("sensor", sensor) == Result::SUCCESS);
    REQUIRE(std::abs(std::any_cast<F32>(sensor.at("gain")) - 1.5f) < 1e-6f);
    REQUIRE(std::any_cast<I16>(sensor.at("count")) == -7);
    REQUIRE(std::any_cast<U32>(sensor.at("index")) == 9);
    REQUIRE(std::any_cast<U64>(sensor.at("big")) == 9223372036854775809ULL);
    REQUIRE(std::any_cast<bool>(sensor.at("flag")) == true);
    REQUIRE(std::any_cast<bool>(sensor.at("off")) == false);

    const auto iq = std::any_cast<CF32>(sensor.at("iq"));
    REQUIRE(std::abs(iq.real() - 1.0f) < 1e-6f);
    REQUIRE(std::abs(iq.imag() - 2.0f) < 1e-6f);
    const auto wide = std::any_cast<CF64>(sensor.at("wide"));
    REQUIRE(std::abs(wide.real() - 3.0) < 1e-9);
    REQUIRE(std::abs(wide.imag() + 4.0) < 1e-9);

    const auto taps = std::any_cast<std::vector<CF32>>(sensor.at("taps"));
    REQUIRE(taps.size() == 2);
    REQUIRE(std::abs(taps[0].real() - 1.0f) < 1e-6f);
    REQUIRE(std::abs(taps[1].imag() + 2.0f) < 1e-6f);
    const auto window = std::any_cast<std::vector<F32>>(sensor.at("window"));
    REQUIRE(window.size() == 2);
    REQUIRE(std::abs(window[0] - 0.25f) < 1e-6f);
    REQUIRE(std::abs(window[1] - 0.75f) < 1e-6f);

    Parser::Map seededOut;
    REQUIRE(flowgraph->environment().get("seeded", seededOut) == Result::SUCCESS);
    REQUIRE(std::abs(std::any_cast<F64>(seededOut.at("level")) - 2.5) < 1e-9);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map report;
    REQUIRE(flowgraph->environment().get("report", report) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(report.at("gain_type")) == "float32");
    REQUIRE(std::any_cast<std::string>(report.at("count_type")) == "int16");
    REQUIRE(std::any_cast<std::string>(report.at("iq_type")) == "complex64");
    REQUIRE(std::any_cast<std::string>(report.at("taps_type")) == "ndarray");
    REQUIRE(std::any_cast<std::string>(report.at("taps_dtype")) == "complex64");
    REQUIRE_FALSE(std::any_cast<bool>(report.at("taps_writable")));
    REQUIRE(std::any_cast<std::string>(report.at("level_type")) == "float64");

    Parser::Map burst;
    REQUIRE(flowgraph->environment().get("burst", burst) == Result::SUCCESS);
    const auto burstWindow = std::any_cast<std::vector<F32>>(burst.at("window"));
    REQUIRE(burstWindow.size() == 4);
    REQUIRE(std::abs(burstWindow[3] - 1.5f) < 1e-6f);
    const auto burstTaps = std::any_cast<std::vector<CF32>>(burst.at("taps"));
    REQUIRE(burstTaps.size() == 2);
    REQUIRE(std::abs(burstTaps[0].real() - 2.0f) < 1e-6f);
    REQUIRE(std::abs(burstTaps[0].imag() - 2.0f) < 1e-6f);
    REQUIRE(std::abs(burstTaps[1].real() - 4.0f) < 1e-6f);
    REQUIRE(std::abs(burstTaps[1].imag() + 4.0f) < 1e-6f);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block subscribes to block metrics",
                 "[modules][python][block]") {
    PythonMetricsSourceTestConfig sourceConfig;
    REQUIRE(flowgraph->blockCreate("peer", sourceConfig, {}, DeviceType::CPU, RuntimeType::NATIVE) ==
            Result::SUCCESS);

    Blocks::Python config;
    config.code =
        "_n = 0\n"
        "\n"
        "def compute(ctx):\n"
        "    global _n\n"
        "    if _n == 0:\n"
        "        ctx.metrics.subscribe_all()\n"
        "        assert \"peer\" not in ctx.metrics\n"
        "        assert ctx.metrics.get(\"ghost\") == {}\n"
        "    else:\n"
        "        assert \"peer\" in ctx.metrics\n"
        "        assert ctx.metrics[\"peer\"].get(\"answer\") == 42\n"
        "        assert \"secret\" not in ctx.metrics[\"peer\"]\n"
        "        assert \"ghost\" in ctx.metrics\n"
        "        assert ctx.metrics[\"ghost\"] == {}\n"
        "    _n += 1\n"
        "    ctx.outputs[0][...] = float(_n)\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[1]"}};

    REQUIRE(flowgraph->blockCreate("python_metrics", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_metrics");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block metrics test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_metrics");
    const Tensor& output = block.outputs.at("output0").tensor;
    REQUIRE(std::abs(output.at<F32>(0) - 3.0f) < 1e-5f);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block exposes console output through module context metric",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    print(\"hello from metric\")\n";
    config.inputCount = 0;
    config.outputCount = 0;

    REQUIRE(flowgraph->blockCreate("python_console", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_console");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block console test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(flowgraph->view().metrics("python_console", block.metrics) == Result::SUCCESS);

    bool sawMetric = false;
    bool sawPrint = false;
    for (const auto& metric : block.metrics) {
        if (metric.format != "private-python-diagnostic" || !metric.value.has_value()) {
            continue;
        }

        sawMetric = true;
        const auto diagnostic = std::any_cast<Runtime::Context::Diagnostic>(metric.value);
        for (const auto& line : diagnostic.console) {
            sawPrint = sawPrint || line.find("hello from metric") != std::string::npos;
        }
    }

    REQUIRE(sawMetric);
    REQUIRE(sawPrint);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block rejects malformed output tensor specs",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    pass\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[1x]"}};

    REQUIRE(flowgraph->blockCreate("python_bad_tensor_spec", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    const auto block = viewBlock("python_bad_tensor_spec");
    REQUIRE(block.state == Block::State::Errored);
}
