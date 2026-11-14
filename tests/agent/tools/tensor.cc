#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "jetstream/agent/tools.hh"
#include "jetstream/detail/block_impl.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"

using namespace Jetstream;
using Catch::Approx;
using Json = nlohmann::json;

namespace {

constexpr U64 kLiveElements = 16384;

Tensor& FixtureOutput() {
    static Tensor tensor;
    return tensor;
}

std::atomic<U64>& LiveFrames() {
    static std::atomic<U64> frames = 0;
    return frames;
}

struct TensorFixtureConfig : Block::Config {
    JST_BLOCK_TYPE(agent_tensor_fixture)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Tensor Fixture", "Exposes a test tensor.", "A test-only block exposing a chosen tensor.")

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

struct TensorFixtureBlock : Block::Impl, DynamicConfig<TensorFixtureConfig> {
    Result define() override {
        return defineInterfaceOutput("out", "Output", "Output tensor.");
    }

    Result create() override {
        if (!FixtureOutput().empty()) {
            outputs()["out"].produced(name(), "out", FixtureOutput());
            outputs()["out"].exposedAs(name(), "out");
        }
        return Result::SUCCESS;
    }
};

struct TensorSinkConfig : Block::Config {
    JST_BLOCK_TYPE(agent_tensor_sink)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Tensor Sink", "Consumes a test tensor.", "A test-only block with one input.")

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

struct TensorSinkBlock : Block::Impl, DynamicConfig<TensorSinkConfig> {
    Result define() override {
        return defineInterfaceInput("in", "Input", "Input tensor.");
    }
};

struct LiveTensorModuleConfig : Module::Config {
    JST_MODULE_TYPE(agent_live_tensor_module)

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

struct LiveTensorModule : Module::Impl,
                          DynamicConfig<LiveTensorModuleConfig>,
                          NativeCpuRuntimeContext,
                          Scheduler::Context {
    Result define() override {
        return defineInterfaceOutput("out");
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::U64, {kLiveElements}));
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        const auto busyUntil = std::chrono::steady_clock::now() + std::chrono::microseconds(500);
        while (std::chrono::steady_clock::now() < busyUntil) {
            std::this_thread::yield();
        }

        const auto frame = LiveFrames().load() + 1;
        auto* values = output.data<U64>();
        for (U64 index = 0; index < kLiveElements; ++index) {
            values[index] = frame;
            if (index == kLiveElements / 2) {
                std::this_thread::yield();
            }
        }
        LiveFrames().store(frame);
        return Result::SUCCESS;
    }

    Tensor output;
};

struct LiveTensorBlockConfig : Block::Config {
    JST_BLOCK_TYPE(agent_live_tensor)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Live Tensor", "Writes frames.", "A test-only block writing a frame counter.")

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

struct LiveTensorBlock : Block::Impl, DynamicConfig<LiveTensorBlockConfig> {
    Result define() override {
        return defineInterfaceOutput("out", "Output", "Output tensor.");
    }

    Result create() override {
        JST_CHECK(moduleCreate("source", std::make_shared<LiveTensorModuleConfig>(), {}));
        JST_CHECK(moduleExposeOutput("out", {"source", "out"}));
        return Result::SUCCESS;
    }
};

struct StallControl {
    std::atomic<bool> stall = false;
    std::atomic<bool> stalled = false;
    std::atomic<U64> frames = 0;
    std::atomic<U64> captures = 0;
    std::atomic<bool> gate = false;
    std::atomic<bool> gated = false;
};

StallControl& Control() {
    static StallControl control;
    return control;
}

struct StallTensorModuleConfig : Module::Config {
    JST_MODULE_TYPE(agent_stall_tensor_module)

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

struct StallTensorModule : Module::Impl,
                           DynamicConfig<StallTensorModuleConfig>,
                           NativeCpuRuntimeContext,
                           Scheduler::Context {
    Result define() override {
        JST_CHECK(defineTaint(Module::Taint::SURFACE));
        return defineInterfaceOutput("out");
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::U64, {64}));
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result presentCapture() override {
        Control().captures += 1;
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        auto& control = Control();
        const auto safety = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        if (control.stall.load()) {
            control.stalled = true;
            while (control.stall.load() && std::chrono::steady_clock::now() < safety) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            control.stalled = false;
        }

        const auto frame = control.frames.load() + 1;
        auto* values = output.data<U64>();
        for (U64 index = 0; index < output.size(); ++index) {
            values[index] = frame;
        }
        control.frames = frame;
        return Result::SUCCESS;
    }

    Tensor output;
};

struct StallTensorBlockConfig : Block::Config {
    JST_BLOCK_TYPE(agent_stall_tensor)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Stall Tensor", "Stalls compute.", "A test-only block with a controllable compute stall.")

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

struct StallTensorBlock : Block::Impl, DynamicConfig<StallTensorBlockConfig> {
    Result define() override {
        return defineInterfaceOutput("out", "Output", "Output tensor.");
    }

    Result create() override {
        JST_CHECK(moduleCreate("source", std::make_shared<StallTensorModuleConfig>(), {}));
        JST_CHECK(moduleExposeOutput("out", {"source", "out"}));
        return Result::SUCCESS;
    }
};

struct GatedTensorBlockConfig : Block::Config {
    JST_BLOCK_TYPE(agent_gated_tensor)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Gated Tensor", "Holds creation.", "A test-only block whose creation waits on a gate.")

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

struct GatedTensorBlock : Block::Impl, DynamicConfig<GatedTensorBlockConfig> {
    Result create() override {
        auto& control = Control();
        const auto safety = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        control.gated = true;
        while (control.gate.load() && std::chrono::steady_clock::now() < safety) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        control.gated = false;
        return Result::SUCCESS;
    }
};

std::atomic<U64>& ProbeCopies() {
    static std::atomic<U64> copies = 0;
    return copies;
}

struct CopyProbe {
    CopyProbe() = default;

    CopyProbe(const CopyProbe&) {
        ProbeCopies() += 1;
    }

    CopyProbe& operator=(const CopyProbe&) {
        ProbeCopies() += 1;
        return *this;
    }
};

JST_REGISTER_MODULE(LiveTensorModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(StallTensorModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_BLOCK(LiveTensorBlock, {"agent_live_tensor_module"});
JST_REGISTER_BLOCK(StallTensorBlock, {"agent_stall_tensor_module"});
JST_REGISTER_BLOCK_NO_MODULES(TensorFixtureBlock);
JST_REGISTER_BLOCK_NO_MODULES(TensorSinkBlock);
JST_REGISTER_BLOCK_NO_MODULES(GatedTensorBlock);

struct NullHost : Agent::Tools::Host {
    bool has(const std::string&) override {
        return false;
    }

    Result create(const std::string&, std::shared_ptr<Flowgraph>&) override {
        return Result::ERROR;
    }
};

Agent::Tools::Output Execute(const std::shared_ptr<Flowgraph>& graph, const std::string& name, const Json& args) {
    NullHost host;
    return Agent::Tools(host).execute(graph, name, args.dump());
}

Json Inspect(const std::shared_ptr<Flowgraph>& graph, const Json& args) {
    return Json::parse(Execute(graph, "inspect_tensor", args).content);
}

Json Selected(const std::string& name, const std::string& port, const std::string& key, const Json& value) {
    Json args = {{"name", name}, {"port", port}};
    args[key] = value;
    return args;
}

struct TensorGraph {
    std::shared_ptr<Flowgraph> graph = std::make_shared<Flowgraph>();

    explicit TensorGraph(const Tensor& tensor) {
        REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
        FixtureOutput() = tensor;
        REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        REQUIRE(graph->blockCreate("source", "agent_tensor_fixture", {}, {}) == Result::SUCCESS);
    }

    ~TensorGraph() {
        FixtureOutput() = Tensor{};
        CHECK(graph->destroy() == Result::SUCCESS);
    }
};

template<typename T>
Tensor MakeTensor(const DataType dtype, const Shape& shape, const std::vector<T>& values) {
    Tensor tensor;
    REQUIRE(tensor.create(DeviceType::CPU, dtype, shape) == Result::SUCCESS);
    REQUIRE(tensor.size() == values.size());
    std::memcpy(tensor.data(), values.data(), values.size() * sizeof(T));
    return tensor;
}

Tensor Ramp(const Shape& shape) {
    U64 size = 1;
    for (const auto extent : shape) {
        size *= extent;
    }
    std::vector<I32> values(size);
    for (U64 index = 0; index < size; ++index) {
        values[index] = static_cast<I32>(index);
    }
    return MakeTensor<I32>(DataType::I32, shape, values);
}

template<typename Condition>
bool WaitFor(Condition&& condition) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!condition()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

struct RunningGraph {
    std::shared_ptr<Flowgraph> graph = std::make_shared<Flowgraph>();
    std::atomic<bool> running = true;
    std::thread compute;
    std::thread present;

    RunningGraph() {
        REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
        auto& control = Control();
        control.stall = false;
        control.frames = 0;
        control.captures = 0;
        REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
        REQUIRE(graph->blockCreate("stall", "agent_stall_tensor", {}, {}) == Result::SUCCESS);
        REQUIRE(graph->start() == Result::SUCCESS);
        compute = std::thread([this] {
            while (running.load()) {
                (void)graph->compute();
            }
        });
        present = std::thread([this] {
            while (running.load()) {
                (void)graph->present();
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });
    }

    ~RunningGraph() {
        Control().stall = false;
        running = false;
        compute.join();
        present.join();
        CHECK(graph->stop() == Result::SUCCESS);
        CHECK(graph->destroy() == Result::SUCCESS);
    }

    bool advances() const {
        const auto frames = Control().frames.load();
        const auto captures = Control().captures.load();
        return WaitFor([&] {
            return Control().frames.load() > frames + 3 && Control().captures.load() > captures + 3;
        });
    }
};

Json Attributes(const Tensor& tensor) {
    TensorGraph fixture(tensor);
    const auto output = Execute(fixture.graph, "inspect_tensor", {{"name", "source"}, {"port", "out"}, {"stats", false}});
    REQUIRE_FALSE(output.error);
    CHECK(output.content.find("inspect_block") == std::string::npos);
    return Json::parse(output.content)["tensor"];
}

constexpr U64 kAttributeBudget = 16384;
constexpr U64 kResponseMargin = 1024;

U64 Serialized(const Json& value) {
    return value.dump(-1, ' ', false, Json::error_handler_t::replace).size();
}

Json BoundedAttributes(const Tensor& tensor) {
    TensorGraph fixture(tensor);
    const auto start = std::chrono::steady_clock::now();
    const auto output = Execute(fixture.graph, "inspect_tensor", {{"name", "source"}, {"port", "out"}, {"stats", false}});
    const auto elapsed = std::chrono::steady_clock::now() - start;
    REQUIRE_FALSE(output.error);
    auto metadata = Json::parse(output.content)["tensor"];
    CHECK(Serialized(metadata["attributes"]) <= kAttributeBudget);
    CHECK(output.content.size() <= kAttributeBudget + kResponseMargin);
    CHECK(elapsed < std::chrono::seconds(2));
    return metadata;
}

std::any Nested(const U64 width, const U64 depth, std::any leaf) {
    for (U64 level = 0; level < depth; ++level) {
        leaf = Parser::Sequence(width, leaf);
    }
    return leaf;
}

std::string EscapedKey(const U64 index) {
    std::string key(64, '\x01');
    key[0] = static_cast<char>(0x01 + (index % 8));
    key[1] = static_cast<char>(0x01 + (index / 8));
    return key;
}

void CheckMarkers(const Json& value) {
    if (value.is_array()) {
        for (const auto& entry : value) {
            CheckMarkers(entry);
        }
    }
    if (!value.is_object()) {
        return;
    }
    if (value.contains("truncated")) {
        CHECK(value["truncated"] == true);
        CHECK(value.contains("type"));
        CHECK(value["size"].get<U64>() > 0);
        if (value["type"] != "string") {
            CHECK(value["preview"].size() <= 32);
            CHECK(value["preview"].size() <= value["size"].get<U64>());
        }
        CheckMarkers(value["preview"]);
        return;
    }
    if (value.contains("omitted")) {
        CHECK(value.contains("type"));
        return;
    }
    for (const auto& [_, entry] : value.items()) {
        CheckMarkers(entry);
    }
}

}  // namespace

TEST_CASE("Tensor inspection rejects malformed arguments", "[agent][tools][tensor]") {
    TensorGraph fixture(Ramp({4}));
    const auto base = [](const std::string& key, const Json& value) {
        return Selected("source", "out", key, value);
    };

    const std::vector<Json> malformed = {
        Json::object(),
        Json{{"name", "source"}},
        Json{{"port", "out"}},
        Json{{"name", 3}, {"port", "out"}},
        Json{{"name", "source"}, {"port", 3}},
        base("direction", "sideways"),
        base("direction", 1),
        base("values", -1),
        base("values", 257),
        base("values", 1.5),
        base("values", "8"),
        base("stats", 1),
        base("unknown", true),
        base("slice", "[0:1]"),
        base("slice", 0),
        base("slice", Json::array({-1})),
        base("slice", Json::array({1.5})),
        base("slice", Json::array({true})),
        base("slice", Json::array({":"})),
        base("slice", Json::array({"...", "..."})),
        base("slice", Json::array({Json{{"step", 0}}})),
        base("slice", Json::array({Json{{"step", -1}}})),
        base("slice", Json::array({Json{{"start", "0"}}})),
        base("slice", Json::array({Json{{"end", 1}}})),
        base("slice", Json(std::vector<int>(33, 0))),
    };
    for (const auto& args : malformed) {
        CAPTURE(args.dump());
        const auto output = Execute(fixture.graph, "inspect_tensor", args);
        CHECK(output.error);
        CHECK(Json::parse(output.content).contains("error"));
    }

    NullHost host;
    CHECK(Agent::Tools(host).execute(fixture.graph, "inspect_tensor", "{").error);

    const auto outOfRange = {
        Json::array({4}),
        Json::array({0, 0}),
        Json::array({Json{{"start", 5}}}),
        Json::array({Json{{"stop", 5}}}),
        Json::array({Json{{"stop", 18446744073709551615ULL}}}),
    };
    for (const auto& slice : outOfRange) {
        CAPTURE(slice.dump());
        const auto body = Inspect(fixture.graph, base("slice", slice));
        CHECK(body.contains("error"));
        CHECK(body["tensor"]["shape"] == Json::array({4}));
        CHECK_FALSE(body.contains("stats"));
    }
}

TEST_CASE("Tensor inspection reports missing graph, block, port, and connection", "[agent][tools][tensor]") {
    CHECK(Execute(nullptr, "inspect_tensor", {{"name", "source"}, {"port", "out"}}).error);

    SECTION("missing block and port") {
        TensorGraph fixture(Ramp({4}));

        const auto block = Execute(fixture.graph, "inspect_tensor", {{"name", "absent"}, {"port", "out"}});
        CHECK(block.error);
        CHECK(Json::parse(block.content)["error"].get<std::string>().find("absent") != std::string::npos);

        const auto port = Execute(fixture.graph, "inspect_tensor", {{"name", "source"}, {"port", "absent"}});
        CHECK(port.error);
        CHECK(Json::parse(port.content)["available_ports"] == Json::array({"out"}));

        const auto direction = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"direction", "input"}});
        CHECK(direction["available_ports"] == Json::array());
    }

    SECTION("disconnected input") {
        TensorGraph fixture(Ramp({4}));

        REQUIRE(fixture.graph->blockCreate("sink", "agent_tensor_sink", {}, {}) != Result::FATAL);

        const auto body = Inspect(fixture.graph, {{"name", "sink"}, {"port", "in"}, {"direction", "input"}});
        CHECK(body["error"].get<std::string>().find("not connected") != std::string::npos);
        CHECK(body["state"] == "incomplete");
        CHECK_FALSE(body.contains("tensor"));
    }

    SECTION("connected input identifies its source and reads upstream data") {
        TensorGraph fixture(Ramp({4}));
        TensorMap inputs;
        inputs["in"].requested("source", "out");
        REQUIRE(fixture.graph->blockCreate("sink", "agent_tensor_sink", {}, inputs) == Result::SUCCESS);

        const auto body = Inspect(fixture.graph, {
            {"name", "sink"}, {"port", "in"}, {"direction", "input"}, {"values", 4},
        });
        REQUIRE_FALSE(body.contains("error"));
        CHECK(body["direction"] == "input");
        CHECK(body["source"] == Json({{"block", "source"}, {"port", "out"}}));
        CHECK(body["values"] == Json::array({0, 1, 2, 3}));
    }

    SECTION("output without a tensor") {
        TensorGraph fixture(Tensor{});

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}});
        CHECK(body["error"].get<std::string>().find("no tensor data") != std::string::npos);
        CHECK(body["state"] == "errored");
    }
}

TEST_CASE("Tensor inspection resolves shape, strides, and offsets", "[agent][tools][tensor]") {
    SECTION("row-major metadata and plain slices") {
        TensorGraph fixture(Ramp({3, 4}));

        const auto whole = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 256}});
        REQUIRE_FALSE(whole.contains("error"));
        CHECK(whole["name"] == "source");
        CHECK(whole["port"] == "out");
        CHECK(whole["direction"] == "output");
        CHECK(whole["tensor"]["dtype"] == "I32");
        CHECK(whole["tensor"]["device"] == "cpu");
        CHECK(whole["tensor"]["native_device"] == "cpu");
        CHECK(whole["tensor"]["shape"] == Json::array({3, 4}));
        CHECK(whole["tensor"]["strides"] == Json::array({4, 1}));
        CHECK(whole["tensor"]["offset"] == 0);
        CHECK(whole["tensor"]["elements"] == 12);
        CHECK(whole["tensor"]["element_size"] == 4);
        CHECK(whole["tensor"]["contiguous"] == true);
        CHECK(whole["values"].size() == 12);
        CHECK(whole["values"][11] == 11);
        CHECK(whole["values_truncated"] == false);

        const auto fixed = Inspect(fixture.graph, Json{
            {"name", "source"}, {"port", "out"}, {"values", 16}, {"slice", Json::array({1})},
        });
        CHECK(fixed["selection"]["shape"] == Json::array({4}));
        CHECK(fixed["selection"]["offset"] == 4);
        CHECK(fixed["values"] == Json::array({4, 5, 6, 7}));

        const auto ranged = Inspect(fixture.graph, Json{
            {"name", "source"}, {"port", "out"}, {"values", 16},
            {"slice", Json::array({Json{{"start", 1}}, Json{{"start", 1}, {"stop", 4}, {"step", 2}}})},
        });
        CHECK(ranged["selection"]["shape"] == Json::array({2, 2}));
        CHECK(ranged["selection"]["strides"] == Json::array({4, 2}));
        CHECK(ranged["selection"]["offset"] == 5);
        CHECK(ranged["values"] == Json::array({5, 7, 9, 11}));
        CHECK(ranged["stats"]["selected_elements"] == 4);
        CHECK(ranged["stats"]["real"]["min"] == 5);
        CHECK(ranged["stats"]["real"]["max"] == 11);

        const auto ellipsis = Inspect(fixture.graph, Json{
            {"name", "source"}, {"port", "out"}, {"values", 16}, {"slice", Json::array({"...", 3})},
        });
        CHECK(ellipsis["values"] == Json::array({3, 7, 11}));

        const auto empty = Inspect(fixture.graph, Json{
            {"name", "source"}, {"port", "out"}, {"values", 16},
            {"slice", Json::array({Json{{"start", 2}, {"stop", 0}}})},
        });
        REQUIRE_FALSE(empty.contains("error"));
        CHECK(empty["selection"]["elements"] == 0);
        CHECK(empty["values"] == Json::array());
        CHECK(empty["stats"]["examined_elements"] == 0);
        CHECK(empty["stats"]["real"]["min"].is_null());
        CHECK(empty["stats"]["power"].is_null());
    }

    SECTION("existing offset and stride are honored") {
        auto view = Ramp({4, 6});
        REQUIRE(view.slice({Token(U64{1}, U64{4}, U64{2}), Token(U64{1}, U64{6}, U64{2})}) == Result::SUCCESS);
        REQUIRE(view.permute({1, 0}) == Result::SUCCESS);
        REQUIRE_FALSE(view.contiguous());
        TensorGraph fixture(view);

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 16}});
        REQUIRE_FALSE(body.contains("error"));
        CHECK(body["tensor"]["shape"] == Json::array({3, 2}));
        CHECK(body["tensor"]["strides"] == Json::array({2, 12}));
        CHECK(body["tensor"]["offset"] == 7);
        CHECK(body["tensor"]["contiguous"] == false);
        CHECK(body["values"] == Json::array({7, 19, 9, 21, 11, 23}));
        CHECK(body["stats"]["real"]["min"] == 7);
        CHECK(body["stats"]["real"]["max"] == 23);
        CHECK(body["stats"]["real"]["mean"].get<F64>() == Approx(15.0));

        const auto nested = Inspect(fixture.graph, Json{
            {"name", "source"}, {"port", "out"}, {"values", 16}, {"slice", Json::array({2, 1})},
        });
        CHECK(nested["selection"]["offset"] == 23);
        CHECK(nested["values"] == Json::array({23}));
    }
}

TEST_CASE("Tensor inspection preserves integer precision", "[agent][tools][tensor]") {
    SECTION("unsigned") {
        const U64 maximum = std::numeric_limits<U64>::max();
        TensorGraph fixture(MakeTensor<U64>(DataType::U64, {2}, {maximum, maximum - 1}));

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 2}});
        CHECK(body["values"][0].get<U64>() == maximum);
        CHECK(body["values"][1].get<U64>() == maximum - 1);
        CHECK(body["stats"]["real"]["max"].get<U64>() == maximum);
        CHECK(body["stats"]["real"]["min"].get<U64>() == maximum - 1);
    }

    SECTION("signed") {
        const I64 minimum = std::numeric_limits<I64>::min();
        TensorGraph fixture(MakeTensor<I64>(DataType::I64, {2}, {minimum, minimum + 1}));

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 2}});
        CHECK(body["values"][0].get<I64>() == minimum);
        CHECK(body["stats"]["real"]["min"].get<I64>() == minimum);
        CHECK(body["stats"]["real"]["max"].get<I64>() == minimum + 1);
    }

    SECTION("narrow complex") {
        TensorGraph fixture(MakeTensor<CI8>(DataType::CI8, {2}, {CI8(-128, 127), CI8(3, -4)}));

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 2}});
        CHECK(body["values"] == Json::array({Json::array({-128, 127}), Json::array({3, -4})}));
        CHECK(body["stats"]["real"]["min"] == -128);
        CHECK(body["stats"]["imag"]["max"] == 127);
        CHECK(body["stats"]["magnitude"]["min"].get<F64>() == Approx(5.0));
    }
}

TEST_CASE("Tensor inspection defines real statistics and non-finite behavior", "[agent][tools][tensor]") {
    const F32 nan = std::numeric_limits<F32>::quiet_NaN();
    const F32 inf = std::numeric_limits<F32>::infinity();

    SECTION("real values") {
        TensorGraph fixture(MakeTensor<F32>(DataType::F32, {4}, {-3.0f, 1.0f, 3.0f, 3.0f}));

        const auto stats = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}})["stats"];
        CHECK(stats["exact"] == true);
        CHECK(stats["sample_step"] == 1);
        CHECK(stats["finite_elements"] == 4);
        CHECK(stats["real"]["min"].get<F64>() == -3.0);
        CHECK(stats["real"]["max"].get<F64>() == 3.0);
        CHECK(stats["real"]["mean"].get<F64>() == Approx(1.0));
        CHECK(stats["power"].get<F64>() == Approx(7.0));
        CHECK(stats["rms"].get<F64>() == Approx(std::sqrt(7.0)));
        CHECK(stats["power_db"].get<F64>() == Approx(10.0 * std::log10(7.0)));
        CHECK_FALSE(stats.contains("imag"));
        CHECK_FALSE(stats.contains("magnitude"));
    }

    SECTION("non-finite values are encoded, counted, and excluded") {
        TensorGraph fixture(MakeTensor<F32>(DataType::F32, {5}, {nan, inf, -inf, 2.0f, 4.0f}));

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 5}});
        CHECK(body["values"] == Json::array({"NaN", "Infinity", "-Infinity", 2.0, 4.0}));
        const auto& stats = body["stats"];
        CHECK(stats["finite_elements"] == 2);
        CHECK(stats["nan_elements"] == 1);
        CHECK(stats["infinite_elements"] == 2);
        CHECK(stats["real"]["min"].get<F64>() == 2.0);
        CHECK(stats["real"]["max"].get<F64>() == 4.0);
        CHECK(stats["real"]["mean"].get<F64>() == Approx(3.0));
        CHECK(stats["power"].get<F64>() == Approx(10.0));
    }

    SECTION("no finite values and zero power") {
        TensorGraph fixture(MakeTensor<F32>(DataType::F32, {2}, {nan, inf}));

        const auto stats = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}})["stats"];
        CHECK(stats["finite_elements"] == 0);
        CHECK(stats["real"]["min"].is_null());
        CHECK(stats["power"].is_null());
        CHECK(stats["power_db"].is_null());
    }

    SECTION("zero power and arithmetic overflow stay explicit") {
        TensorGraph zeros(MakeTensor<F32>(DataType::F32, {2}, {0.0f, 0.0f}));
        const auto zero = Inspect(zeros.graph, {{"name", "source"}, {"port", "out"}})["stats"];
        CHECK(zero["power"].get<F64>() == 0.0);
        CHECK(zero["power_db"] == "-Infinity");
    }

    SECTION("wide values do not overflow the mean") {
        const F64 huge = std::numeric_limits<F64>::max();
        TensorGraph fixture(MakeTensor<F64>(DataType::F64, {2}, {huge, huge}));

        const auto stats = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}})["stats"];
        CHECK(stats["real"]["mean"].get<F64>() == Approx(huge));
        CHECK(stats["power"] == "Infinity");
    }
}

TEST_CASE("Tensor inspection defines complex statistics", "[agent][tools][tensor]") {
    const F32 nan = std::numeric_limits<F32>::quiet_NaN();
    const F32 inf = std::numeric_limits<F32>::infinity();

    SECTION("finite samples") {
        TensorGraph fixture(MakeTensor<CF32>(DataType::CF32, {3}, {CF32(3.0f, 4.0f), CF32(-1.0f, 0.0f), CF32(0.0f, -2.0f)}));

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 3}});
        CHECK(body["values"] == Json::array({Json::array({3.0, 4.0}), Json::array({-1.0, 0.0}), Json::array({0.0, -2.0})}));
        const auto& stats = body["stats"];
        CHECK(stats["real"]["min"].get<F64>() == -1.0);
        CHECK(stats["real"]["max"].get<F64>() == 3.0);
        CHECK(stats["real"]["mean"].get<F64>() == Approx(2.0 / 3.0));
        CHECK(stats["imag"]["min"].get<F64>() == -2.0);
        CHECK(stats["imag"]["max"].get<F64>() == 4.0);
        CHECK(stats["imag"]["mean"].get<F64>() == Approx(2.0 / 3.0));
        CHECK(stats["magnitude"]["min"].get<F64>() == Approx(1.0));
        CHECK(stats["magnitude"]["max"].get<F64>() == Approx(5.0));
        CHECK(stats["power"].get<F64>() == Approx(10.0));
        CHECK(stats["rms"].get<F64>() == Approx(std::sqrt(10.0)));
        CHECK(stats["power_db"].get<F64>() == Approx(10.0));
    }

    SECTION("one non-finite part excludes the whole element") {
        TensorGraph fixture(MakeTensor<CF32>(DataType::CF32, {4}, {
            CF32(nan, 100.0f), CF32(100.0f, inf), CF32(-inf, nan), CF32(3.0f, 4.0f),
        }));

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 4}});
        CHECK(body["values"][0] == Json::array({"NaN", 100.0}));
        CHECK(body["values"][1] == Json::array({100.0, "Infinity"}));
        const auto& stats = body["stats"];
        CHECK(stats["finite_elements"] == 1);
        CHECK(stats["nan_elements"] == 2);
        CHECK(stats["infinite_elements"] == 1);
        CHECK(stats["imag"]["max"].get<F64>() == 4.0);
        CHECK(stats["power"].get<F64>() == Approx(25.0));
    }

    SECTION("complex channelized layout from a stridden view") {
        std::vector<CF32> samples;
        for (U64 channel = 0; channel < 2; ++channel) {
            for (U64 sample = 0; sample < 3; ++sample) {
                samples.emplace_back(static_cast<F32>(channel), static_cast<F32>(sample));
            }
        }
        auto view = MakeTensor<CF32>(DataType::CF32, {2, 3}, samples);
        REQUIRE(view.permute({1, 0}) == Result::SUCCESS);
        TensorGraph fixture(view);

        const auto body = Inspect(fixture.graph, Json{
            {"name", "source"}, {"port", "out"}, {"values", 2}, {"slice", Json::array({2})},
        });
        CHECK(body["values"] == Json::array({Json::array({0.0, 2.0}), Json::array({1.0, 2.0})}));
        CHECK(body["stats"]["imag"]["min"].get<F64>() == 2.0);
        CHECK(body["stats"]["real"]["max"].get<F64>() == 1.0);
    }
}

TEST_CASE("Tensor inspection enforces response and work limits", "[agent][tools][tensor]") {
    const U64 cap = U64{1} << 20;

    SECTION("preview cap") {
        TensorGraph fixture(Ramp({512}));

        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 256}});
        CHECK(body["values"].size() == 256);
        CHECK(body["values"][255] == 255);
        CHECK(body["values_truncated"] == true);
        CHECK(body["stats"]["exact"] == true);
        CHECK(body["stats"]["real"]["max"] == 511);

        const auto rejected = Execute(fixture.graph, "inspect_tensor", {{"name", "source"}, {"port", "out"}, {"values", 257}});
        CHECK(rejected.error);
    }

    SECTION("statistics are exact at the cap") {
        Tensor tensor;
        REQUIRE(tensor.create(DeviceType::CPU, DataType::U8, {cap}) == Result::SUCCESS);
        tensor.data<U8>()[cap - 1] = 200;
        TensorGraph fixture(tensor);

        const auto stats = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}})["stats"];
        CHECK(stats["exact"] == true);
        CHECK(stats["examined_elements"] == cap);
        CHECK(stats["real"]["max"] == 200);
    }

    SECTION("statistics above the cap identify their coverage") {
        Tensor tensor;
        REQUIRE(tensor.create(DeviceType::CPU, DataType::U8, {cap + 1}) == Result::SUCCESS);
        tensor.data<U8>()[1] = 200;
        tensor.data<U8>()[cap] = 100;
        TensorGraph fixture(tensor);

        const auto sampled = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}})["stats"];
        CHECK(sampled["exact"] == false);
        CHECK(sampled["selected_elements"] == cap + 1);
        CHECK(sampled["sample_step"] == 2);
        CHECK(sampled["examined_elements"] == cap / 2 + 1);
        CHECK(sampled["real"]["max"] == 100);

        const auto narrowed = Inspect(fixture.graph, Json{
            {"name", "source"}, {"port", "out"}, {"slice", Json::array({Json{{"stop", 4}}})},
        })["stats"];
        CHECK(narrowed["exact"] == true);
        CHECK(narrowed["real"]["max"] == 200);
    }

    SECTION("the workload shape stays bounded without allocating or scanning the full size") {
        auto tensor = MakeTensor<CF32>(DataType::CF32, {2}, {CF32(3.0f, 4.0f), CF32(0.0f, 0.0f)});
        REQUIRE(tensor.broadcastTo({28, 192, 8192, 2}) == Result::SUCCESS);
        TensorGraph fixture(tensor);

        const auto start = std::chrono::steady_clock::now();
        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"values", 4}});
        const auto elapsed = std::chrono::steady_clock::now() - start;

        REQUIRE_FALSE(body.contains("error"));
        const U64 elements = U64{28} * 192 * 8192 * 2;
        CHECK(body["tensor"]["elements"] == elements);
        CHECK(body["values"].size() == 4);
        CHECK(body["values_truncated"] == true);
        CHECK(body["stats"]["exact"] == false);
        CHECK(body["stats"]["selected_elements"] == elements);
        CHECK(body["stats"]["examined_elements"].get<U64>() <= cap);
        CHECK(body["stats"]["sample_step"] == 85);
        CHECK(body["stats"]["magnitude"]["max"].get<F64>() == Approx(5.0));
        CHECK(body["stats"]["magnitude"]["min"].get<F64>() == 0.0);
        CHECK(body["stats"]["power"].get<F64>() == Approx(12.5).margin(0.01));
        CHECK(body.dump().size() < 2048);
        CHECK(elapsed < std::chrono::seconds(2));
    }

    SECTION("attributes are bounded") {
        auto tensor = Ramp({1});
        for (U64 index = 0; index < 40; ++index) {
            REQUIRE(tensor.setAttribute("key" + std::to_string(100 + index), std::string("value")) == Result::SUCCESS);
        }
        TensorGraph fixture(tensor);

        const auto metadata = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}})["tensor"];
        CHECK(metadata["attributes"].size() == 32);
        CHECK(metadata["attributes_omitted"] == 8);
    }
}

TEST_CASE("Tensor inspection leaves graph, data, and attributes unchanged", "[agent][tools][tensor]") {
    auto tensor = Ramp({8, 8});
    REQUIRE(tensor.setAttribute("sample_rate", F64{2e6}) == Result::SUCCESS);
    REQUIRE(tensor.setDerivedAttribute("label", [] { return std::any(std::string("derived")); }) == Result::SUCCESS);
    const auto id = tensor.id();
    const auto shape = tensor.shape();
    const auto stride = tensor.stride();
    const auto offset = tensor.offset();
    const auto keys = tensor.attributeKeys();
    const std::vector<I32> before(tensor.data<I32>(), tensor.data<I32>() + tensor.size());
    TensorGraph fixture(tensor);

    const auto graphBefore = Execute(fixture.graph, "inspect_flowgraph", {{"detail", "full"}}).content;

    const auto body = Inspect(fixture.graph, Json{
        {"name", "source"}, {"port", "out"}, {"values", 8},
        {"slice", Json::array({Json{{"start", 1}, {"stop", 7}, {"step", 2}}, 3})},
    });
    REQUIRE_FALSE(body.contains("error"));
    CHECK(body["tensor"]["attributes"]["sample_rate"].get<F64>() == 2e6);
    CHECK(body["tensor"]["attributes"]["label"] == "derived");
    CHECK(body["selection"]["shape"] == Json::array({3}));
    CHECK(Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}})["tensor"]["shape"] == Json::array({8, 8}));

    CHECK(Execute(fixture.graph, "inspect_flowgraph", {{"detail", "full"}}).content == graphBefore);
    CHECK(tensor.id() == id);
    CHECK(tensor.shape() == shape);
    CHECK(tensor.stride() == stride);
    CHECK(tensor.offset() == offset);
    CHECK(tensor.attributeKeys() == keys);
    CHECK(std::vector<I32>(tensor.data<I32>(), tensor.data<I32>() + tensor.size()) == before);
}

TEST_CASE("Tensor inspection does not dereference device memory", "[agent][tools][tensor]") {
    Tensor device;
    if (device.create(DeviceType::Metal, DataType::F32, {4}) != Result::SUCCESS &&
        device.create(DeviceType::Vulkan, DataType::F32, {4}) != Result::SUCCESS &&
        device.create(DeviceType::CUDA, DataType::F32, {4}) != Result::SUCCESS) {
        SKIP("No non-CPU device is available in this build.");
    }
    TensorGraph fixture(device);

    const auto rejected = Execute(fixture.graph, "inspect_tensor", {{"name", "source"}, {"port", "out"}, {"values", 1}});
    CHECK(rejected.error);
    const auto body = Json::parse(rejected.content);
    CHECK(body["code"] == "UNSUPPORTED_DEVICE");
    CHECK(body["tensor"]["shape"] == Json::array({4}));
    CHECK_FALSE(body.contains("values"));
    CHECK_FALSE(body.contains("stats"));

    const auto metadata = Execute(fixture.graph, "inspect_tensor", {{"name", "source"}, {"port", "out"}, {"stats", false}});
    CHECK_FALSE(metadata.error);
    CHECK(Json::parse(metadata.content)["tensor"]["dtype"] == "F32");
}

TEST_CASE("Block inspection reports real module timing", "[agent][tools][inspect]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    LiveFrames() = 0;
    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("live", "agent_live_tensor", {}, {}) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("taps", "filter_taps", {}, {}) == Result::SUCCESS);

    const auto timingOf = [&](const std::string& name) {
        const auto body = Json::parse(
            Execute(graph, "inspect_block", {{"name", name}, {"include_config", false}}).content);
        for (const auto& metric : body["metrics"]) {
            if (metric["name"].get<std::string>().starts_with("runtime:")) {
                return metric["value"];
            }
        }
        return Json{};
    };

    const auto idle = timingOf("live");
    CHECK(idle["cycles"] == 0);
    CHECK_FALSE(idle["runtime"].get<std::string>().empty());
    CHECK_FALSE(idle.contains("last_compute_ms"));
    CHECK_FALSE(idle.contains("cycles_per_second"));

    REQUIRE(graph->start() == Result::SUCCESS);
    for (U64 cycle = 0; cycle < 5; ++cycle) {
        REQUIRE(graph->compute() == Result::SUCCESS);
    }

    const auto live = timingOf("live");
    CHECK(live["cycles"].get<U64>() >= 2);
    CHECK(live["last_compute_ms"].get<F64>() >= 0.0);
    CHECK(live["average_compute_ms"].get<F64>() >= 0.0);
    CHECK(live["cycles_per_second"].get<F64>() > 0.0);
    CHECK(live["last_cycle_age_ms"].get<F64>() >= 0.0);
    CHECK_FALSE(live.contains("settled"));

    const auto taps = timingOf("taps");
    CHECK(taps["cycles"] == 1);
    CHECK(taps["settled"] == true);
    CHECK(taps.contains("last_compute_ms"));
    CHECK_FALSE(taps.contains("average_compute_ms"));
    CHECK_FALSE(taps.contains("cycles_per_second"));

    REQUIRE(graph->stop() == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Tensor inspection snapshots do not tear against running compute", "[agent][tools][tensor]") {
    REQUIRE(Registry::DrainStaticRegistrations() == Result::SUCCESS);
    LiveFrames() = 0;
    auto graph = std::make_shared<Flowgraph>();
    REQUIRE(graph->create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(graph->blockCreate("live", "agent_live_tensor", {}, {}) == Result::SUCCESS);
    REQUIRE(graph->start() == Result::SUCCESS);

    std::atomic<bool> running = true;
    std::thread compute([&] {
        while (running.load()) {
            (void)graph->compute();
        }
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    U64 reads = 0;
    U64 torn = 0;
    U64 regressed = 0;
    U64 errors = 0;
    U64 distinct = 0;
    U64 previous = 0;
    auto slowest = std::chrono::steady_clock::duration::zero();
    while (reads < 400 && std::chrono::steady_clock::now() < deadline) {
        const auto start = std::chrono::steady_clock::now();
        const auto body = Inspect(graph, {{"name", "live"}, {"port", "out"}});
        slowest = std::max(slowest, std::chrono::steady_clock::now() - start);
        reads += 1;
        if (body.contains("error")) {
            errors += 1;
            continue;
        }
        const auto minimum = body["stats"]["real"]["min"].get<U64>();
        const auto maximum = body["stats"]["real"]["max"].get<U64>();
        torn += minimum != maximum;
        regressed += minimum < previous;
        distinct += minimum != previous;
        previous = minimum;
    }

    running = false;
    compute.join();

    CHECK(errors == 0);
    CHECK(torn == 0);
    CHECK(regressed == 0);
    CHECK(reads >= 400);
    CHECK(distinct > 1);
    CHECK(slowest < std::chrono::seconds(1));
    CHECK(LiveFrames().load() > 1);

    REQUIRE(graph->stop() == Result::SUCCESS);
    REQUIRE(graph->destroy() == Result::SUCCESS);
}

TEST_CASE("Tensor inspection times out instead of waiting for a long compute segment", "[agent][tools][tensor]") {
    RunningGraph fixture;
    auto& control = Control();
    const Json args = {{"name", "stall"}, {"port", "out"}, {"values", 2}};

    REQUIRE(fixture.advances());
    const auto healthy = Inspect(fixture.graph, args);
    REQUIRE_FALSE(healthy.contains("error"));
    CHECK(healthy["stats"]["real"]["min"] == healthy["stats"]["real"]["max"]);

    control.stall = true;
    REQUIRE(WaitFor([&] { return control.stalled.load(); }));
    const auto frames = control.frames.load();
    const auto graphBefore = Execute(fixture.graph, "inspect_flowgraph", {{"detail", "full"}}).content;

    for (U64 attempt = 0; attempt < 2; ++attempt) {
        const auto start = std::chrono::steady_clock::now();
        const auto output = Execute(fixture.graph, "inspect_tensor", args);
        const auto elapsed = std::chrono::steady_clock::now() - start;

        CHECK(output.error);
        const auto body = Json::parse(output.content);
        CHECK(body["code"] == "TENSOR_BUSY");
        CHECK(body["retriable"] == true);
        CHECK(body["name"] == "stall");
        CHECK(body["port"] == "out");
        CHECK(body["error"].get<std::string>().find("not found") == std::string::npos);
        CHECK_FALSE(body.contains("tensor"));
        CHECK_FALSE(body.contains("selection"));
        CHECK_FALSE(body.contains("stats"));
        CHECK_FALSE(body.contains("values"));
        CHECK(elapsed >= std::chrono::milliseconds(450));
        CHECK(elapsed < std::chrono::milliseconds(1500));
    }

    CHECK(control.stalled.load());
    CHECK(control.frames.load() == frames);
    CHECK(Execute(fixture.graph, "inspect_flowgraph", {{"detail", "full"}}).content == graphBefore);

    control.stall = false;
    CHECK(fixture.advances());

    const auto recovered = Inspect(fixture.graph, args);
    REQUIRE_FALSE(recovered.contains("error"));
    CHECK(recovered["stats"]["real"]["min"] == recovered["stats"]["real"]["max"]);
    CHECK(recovered["stats"]["real"]["min"].get<U64>() > frames);
    CHECK(recovered["values"][0] == recovered["values"][1]);
}

TEST_CASE("Tensor reads release compute when the reader throws", "[agent][tools][tensor]") {
    RunningGraph fixture;
    REQUIRE(fixture.advances());

    bool invoked = false;
    CHECK_THROWS_AS(fixture.graph->view().readTensor("stall", "out", true, [&](const TensorLink&) -> Result {
        invoked = true;
        throw std::runtime_error("reader failed");
    }), std::runtime_error);
    CHECK(invoked);

    CHECK(fixture.advances());
    CHECK_FALSE(Inspect(fixture.graph, {{"name", "stall"}, {"port", "out"}}).contains("error"));
}

TEST_CASE("Tensor inspection fails fast while the graph is mutating", "[agent][tools][tensor]") {
    TensorGraph fixture(Ramp({4}));
    auto& control = Control();
    control.gate = true;
    std::thread mutation([&] {
        (void)fixture.graph->blockCreate("gated", "agent_gated_tensor", {}, {});
    });
    const auto opened = WaitFor([&] { return control.gated.load(); });

    const Json args = {{"name", "source"}, {"port", "out"}, {"values", 1}};
    const auto start = std::chrono::steady_clock::now();
    const auto output = Execute(fixture.graph, "inspect_tensor", args);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    control.gate = false;
    mutation.join();

    REQUIRE(opened);
    CHECK(output.error);
    const auto body = Json::parse(output.content);
    CHECK(body["code"] == "TENSOR_BUSY");
    CHECK(body["retriable"] == true);
    CHECK_FALSE(body.contains("tensor"));
    CHECK(elapsed < std::chrono::milliseconds(250));

    CHECK_FALSE(Inspect(fixture.graph, args).contains("error"));
}

TEST_CASE("Tensor attributes keep ordinary values", "[agent][tools][tensor]") {
    auto tensor = Ramp({1});
    Parser::Map small;
    small["count"] = U64{3};
    small["items"] = Parser::Sequence{std::string("a"), F64{1.5}, std::any()};
    const std::string edge(256, 'x');
    REQUIRE(tensor.setAttribute("sampleRate", F64{2e6}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("frequency", F32{1.5f}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("nan", std::numeric_limits<F32>::quiet_NaN()) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("flag", true) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("offset", I32{-4}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("huge", std::numeric_limits<U64>::max()) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("station", std::string("alpha")) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("edge", edge) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("phase", CF32(1.0f, -2.0f)) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("offsets", std::vector<F32>{1.0f, 2.0f}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("full", std::vector<U64>(32, 7)) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("flags", std::vector<bool>{true, false}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("names", std::vector<std::string>{"a", "b"}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("range", Range<F32>{1.0f, 2.0f}) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("small", small) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("empty", std::any()) == Result::SUCCESS);
    REQUIRE(tensor.setAttribute("probe", CopyProbe{}) == Result::SUCCESS);
    REQUIRE(tensor.setDerivedAttribute("label", [] { return std::any(std::string("derived")); }) == Result::SUCCESS);
    REQUIRE(tensor.setDerivedAttribute("rate", [] { return std::any(F32{48000.0f}); }) == Result::SUCCESS);

    const auto metadata = Attributes(tensor);
    const auto& attributes = metadata["attributes"];
    CHECK_FALSE(metadata.contains("attributes_omitted"));
    CHECK(attributes["sampleRate"].get<F64>() == 2e6);
    CHECK(attributes["frequency"].get<F64>() == 1.5);
    CHECK(attributes["nan"] == "NaN");
    CHECK(attributes["flag"] == true);
    CHECK(attributes["offset"] == -4);
    CHECK(attributes["huge"].get<U64>() == std::numeric_limits<U64>::max());
    CHECK(attributes["station"] == "alpha");
    CHECK(attributes["edge"] == edge);
    CHECK(attributes["phase"] == Json::array({1.0, -2.0}));
    CHECK(attributes["offsets"] == Json::array({1.0, 2.0}));
    CHECK(attributes["full"].size() == 32);
    CHECK(attributes["flags"] == Json::array({true, false}));
    CHECK(attributes["names"] == Json::array({"a", "b"}));
    CHECK(attributes["range"].is_string());
    CHECK(attributes["small"]["count"] == 3);
    CHECK(attributes["small"]["items"] == Json::array({"a", 1.5, nullptr}));
    CHECK(attributes["empty"].is_null());
    CHECK(attributes["probe"] == Json({{"omitted", "unsupported"}, {"type", "value"}}));
    CHECK(attributes["label"] == "derived");
    CHECK(attributes["rate"].get<F64>() == 48000.0);
}

TEST_CASE("Tensor attributes are bounded before they are materialized", "[agent][tools][tensor]") {
    SECTION("stored values are never copied") {
        Parser::Map nested;
        nested["probe"] = CopyProbe{};
        nested["items"] = Parser::Sequence{CopyProbe{}, std::any(nested)};
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("object", nested) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("probe", CopyProbe{}) == Result::SUCCESS);
        TensorGraph fixture(tensor);

        ProbeCopies() = 0;
        const auto body = Inspect(fixture.graph, {{"name", "source"}, {"port", "out"}, {"stats", false}});
        CHECK(ProbeCopies().load() == 0);
        CHECK(body["tensor"]["attributes"]["object"]["probe"]["omitted"] == "unsupported");
    }

    SECTION("large vector") {
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("large", std::vector<F64>(4'000'000, 1.5)) == Result::SUCCESS);

        const auto metadata = Attributes(tensor);
        const auto& large = metadata["attributes"]["large"];
        CHECK(large["truncated"] == true);
        CHECK(large["type"] == "array");
        CHECK(large["size"] == 4'000'000);
        CHECK(large["preview"].size() == 32);
        CHECK(large["preview"][31].get<F64>() == 1.5);
        CHECK(metadata.dump().size() < 2048);
    }

    SECTION("large and wide object") {
        Parser::Map large;
        large.reserve(100'000);
        for (U64 index = 0; index < 100'000; ++index) {
            large["key" + std::to_string(100'000 + index)] = U64{1};
        }
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("large", large) == Result::SUCCESS);

        const auto metadata = Attributes(tensor);
        const auto& summary = metadata["attributes"]["large"];
        CHECK(summary["truncated"] == true);
        CHECK(summary["type"] == "object");
        CHECK(summary["size"] == 100'000);
        CHECK(summary["preview"].size() == 32);
        CHECK(metadata.dump().size() < 2048);
    }

    SECTION("deep nesting") {
        std::any sequence = U64{1};
        std::any map = U64{1};
        for (U64 depth = 0; depth < 64; ++depth) {
            sequence = Parser::Sequence{sequence};
            Parser::Map level;
            level["next"] = map;
            map = level;
        }
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("sequence", sequence) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("map", map) == Result::SUCCESS);

        const auto metadata = Attributes(tensor);
        const auto& attributes = metadata["attributes"];
        CHECK(attributes["sequence"][0][0][0][0] == Json({{"omitted", "depth"}, {"type", "array"}, {"size", 1}}));
        CHECK(attributes["map"]["next"]["next"]["next"]["next"] ==
              Json({{"omitted", "depth"}, {"type", "object"}, {"size", 1}}));
        CHECK(metadata.dump().size() < 2048);
    }

    SECTION("long strings and keys") {
        std::string accented;
        for (U64 index = 0; index < 100'000; ++index) {
            accented += "\xC3\xA9";
        }
        const std::string longKey(65, 'k');
        const std::string edgeKey(64, 'e');
        Parser::Map nested;
        nested[longKey] = U64{1};
        nested["kept"] = U64{2};
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("text", accented) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("nested", nested) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute(longKey, U64{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute(edgeKey, U64{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("bytes", std::string("\xFF\xFE")) == Result::SUCCESS);

        const auto metadata = Attributes(tensor);
        const auto& attributes = metadata["attributes"];
        CHECK(attributes["text"]["truncated"] == true);
        CHECK(attributes["text"]["type"] == "string");
        CHECK(attributes["text"]["size"] == 200'000);
        const auto preview = attributes["text"]["preview"].get<std::string>();
        CHECK(preview.size() == 256);
        CHECK(preview.find("\xEF\xBF\xBD") == std::string::npos);

        CHECK(attributes["nested"]["truncated"] == true);
        CHECK(attributes["nested"]["size"] == 2);
        CHECK(attributes["nested"]["preview"] == Json({{"kept", 2}}));

        CHECK(attributes.contains(edgeKey));
        CHECK_FALSE(attributes.contains(longKey));
        CHECK(metadata["attributes_omitted"] == 1);
        CHECK(attributes["bytes"].is_string());
        CHECK(metadata.dump().size() < 2048);
    }

    SECTION("total output stays within budget") {
        Parser::Map wide;
        for (U64 index = 0; index < 64; ++index) {
            wide["key" + std::to_string(100 + index)] = std::string(256, '\x01');
        }
        auto tensor = Ramp({1});
        for (U64 index = 0; index < 30; ++index) {
            REQUIRE(tensor.setAttribute("attribute" + std::to_string(100 + index), wide) == Result::SUCCESS);
        }
        REQUIRE(tensor.setAttribute("zrate", F64{2e6}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("zstation", std::string("alpha")) == Result::SUCCESS);
        for (U64 index = 0; index < 4; ++index) {
            REQUIRE(tensor.setAttribute("zz" + std::to_string(index), U64{1}) == Result::SUCCESS);
        }

        const auto start = std::chrono::steady_clock::now();
        const auto metadata = Attributes(tensor);
        const auto elapsed = std::chrono::steady_clock::now() - start;

        CHECK(Serialized(metadata["attributes"]) <= kAttributeBudget);
        CHECK(metadata["attributes"].size() == 32);
        CHECK(metadata["attributes_omitted"] == 4);
        CHECK(metadata["attributes"]["attribute100"]["truncated"] == true);
        CHECK(metadata["attributes"]["attribute129"]["truncated"] == true);
        CHECK(metadata["attributes"]["attribute129"]["size"] == 64);
        CHECK(metadata["attributes"]["zrate"].get<F64>() == 2e6);
        CHECK(metadata["attributes"]["zstation"] == "alpha");
        CHECK(elapsed < std::chrono::seconds(2));
    }

    SECTION("nested empty arrays stay within the complete budget") {
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("nested_arrays", Nested(8, 4, Parser::Sequence{})) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("sampleRate", F64{48000.0}) == Result::SUCCESS);

        const auto metadata = BoundedAttributes(tensor);
        const auto& attributes = metadata["attributes"];
        CHECK(attributes["sampleRate"].get<F64>() == 48000.0);
        CHECK(attributes["nested_arrays"].size() == 8);
        CHECK(attributes["nested_arrays"][7][7][7][7] == Json::array());
    }

    SECTION("depth markers cannot multiply past the budget") {
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("markers", Nested(32, 4, Parser::Sequence{U64{1}})) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("sampleRate", F64{48000.0}) == Result::SUCCESS);

        const auto metadata = BoundedAttributes(tensor);
        const auto& attributes = metadata["attributes"];
        CHECK(attributes["sampleRate"].get<F64>() == 48000.0);
        CHECK(attributes["markers"]["truncated"] == true);
        CHECK(attributes["markers"]["size"] == 32);
        CHECK(attributes["markers"]["preview"][0]["preview"][0]["preview"][0][0] ==
              Json({{"omitted", "depth"}, {"type", "array"}, {"size", 1}}));
        CHECK(Serialized(attributes) > kAttributeBudget - 1024);
        CheckMarkers(attributes);
    }

    SECTION("containers without primitives are charged") {
        Parser::Map empties;
        for (U64 index = 0; index < 32; ++index) {
            empties["k" + std::to_string(100 + index)] = Parser::Map{};
        }
        Parser::Map wide;
        for (U64 index = 0; index < 32; ++index) {
            wide["k" + std::to_string(100 + index)] = Nested(32, 2, empties);
        }
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("arrays", Nested(32, 3, Parser::Sequence{})) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("maps", Nested(32, 2, empties)) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("object", wide) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("vectors", std::vector<Parser::Map>(32, empties)) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("zrate", F64{2e6}) == Result::SUCCESS);

        const auto metadata = BoundedAttributes(tensor);
        const auto& attributes = metadata["attributes"];
        CHECK(attributes["zrate"].get<F64>() == 2e6);
        CHECK(attributes["arrays"].is_object());
        CHECK(attributes["maps"].is_object());
        CHECK(attributes["object"]["truncated"] == true);
        CHECK(attributes["object"]["preview"].size() > 0);
        CHECK(attributes["vectors"].is_object());
        CheckMarkers(attributes);
    }

    SECTION("repeated bulky attributes share the budget and keep metadata") {
        const auto bulky = Nested(16, 4, Parser::Sequence{std::string(256, '\x01')});
        auto tensor = Ramp({1});
        for (U64 index = 0; index < 28; ++index) {
            REQUIRE(tensor.setAttribute("attribute" + std::to_string(100 + index), bulky) == Result::SUCCESS);
        }
        REQUIRE(tensor.setAttribute("sampleRate", F64{48000.0}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("station", std::string("alpha")) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("zflag", true) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("zphase", CF64(1.0, -2.0)) == Result::SUCCESS);
        const auto keys = tensor.attributeKeys();

        const auto metadata = BoundedAttributes(tensor);
        const auto& attributes = metadata["attributes"];
        CHECK(attributes.size() == 32);
        CHECK_FALSE(metadata.contains("attributes_omitted"));
        CHECK(attributes["sampleRate"].get<F64>() == 48000.0);
        CHECK(attributes["station"] == "alpha");
        CHECK(attributes["zflag"] == true);
        CHECK(attributes["zphase"] == Json::array({1.0, -2.0}));
        CHECK(attributes["attribute100"]["truncated"] == true);
        CHECK(attributes["attribute100"]["preview"].size() > 0);
        CHECK(attributes["attribute127"]["truncated"] == true);
        CHECK(attributes["attribute127"]["preview"].size() > 0);
        CheckMarkers(attributes);

        CHECK(tensor.attributeKeys() == keys);
        CHECK(std::any_cast<F64>(tensor.attribute("sampleRate")) == 48000.0);
        CHECK(std::any_cast<Parser::Sequence>(tensor.attribute("attribute100")).size() == 16);
    }

    SECTION("worst case escaped keys and strings stay within budget") {
        Parser::Map escaped;
        for (U64 index = 0; index < 32; ++index) {
            escaped[EscapedKey(index)] = std::string(256, '\x01');
        }
        auto tensor = Ramp({1});
        for (U64 index = 0; index < 32; ++index) {
            const auto key = EscapedKey(index);
            if (index % 3 == 0) {
                REQUIRE(tensor.setAttribute(key, std::string(256, '\x01')) == Result::SUCCESS);
            } else if (index % 3 == 1) {
                REQUIRE(tensor.setAttribute(key, escaped) == Result::SUCCESS);
            } else {
                REQUIRE(tensor.setAttribute(key, F64{1.5}) == Result::SUCCESS);
            }
        }

        const auto metadata = BoundedAttributes(tensor);
        const auto& attributes = metadata["attributes"];
        CHECK(attributes.size() == 32);
        for (U64 index = 0; index < 32; ++index) {
            const auto key = EscapedKey(index);
            REQUIRE(attributes.contains(key));
            if (index % 3 == 2) {
                CHECK(attributes[key].get<F64>() == 1.5);
            } else {
                CHECK(attributes[key]["truncated"] == true);
            }
        }
        CheckMarkers(attributes);
    }

    SECTION("every scalar survives the widest keys and values") {
        const F64 widest = -std::numeric_limits<F64>::max();
        auto tensor = Ramp({1});
        for (U64 index = 0; index < 32; ++index) {
            REQUIRE(tensor.setAttribute(EscapedKey(index), CF64(widest, widest)) == Result::SUCCESS);
        }

        const auto metadata = BoundedAttributes(tensor);
        for (U64 index = 0; index < 32; ++index) {
            CHECK(metadata["attributes"][EscapedKey(index)] == Json::array({widest, widest}));
        }
    }

    SECTION("long strings keep truthful summaries under pressure") {
        auto tensor = Ramp({1});
        for (U64 index = 0; index < 32; ++index) {
            REQUIRE(tensor.setAttribute("text" + std::to_string(100 + index), std::string(100'000, '"')) == Result::SUCCESS);
        }

        const auto metadata = BoundedAttributes(tensor);
        const auto& attributes = metadata["attributes"];
        for (const auto* key : {"text100", "text131"}) {
            CHECK(attributes[key]["truncated"] == true);
            CHECK(attributes[key]["size"] == 100'000);
            const auto preview = attributes[key]["preview"].get<std::string>();
            CHECK(preview.size() > 0);
            CHECK(preview.size() <= 256);
        }
        CheckMarkers(attributes);
    }

    SECTION("empty containers at the depth limit stay exact") {
        auto tensor = Ramp({1});
        REQUIRE(tensor.setAttribute("sequence", Nested(1, 4, Parser::Sequence{})) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("object", Nested(1, 4, Parser::Map{})) == Result::SUCCESS);

        const auto attributes = BoundedAttributes(tensor)["attributes"];
        CHECK(attributes["sequence"][0][0][0][0] == Json::array());
        CHECK(attributes["object"][0][0][0][0] == Json::object());
    }
}
