#include <catch2/catch_test_macros.hpp>

#include "compositor/default/presenters/flowgraph/config.hh"
#include "compositor/default/views/flowgraph/editor/config/base.hh"

#include <string>
#include <vector>

using namespace Jetstream;

TEST_CASE("Shared config presenter preserves descriptors values and Python diagnostics", "[core][sakura][interface]") {
    Flowgraph::View::BlockData block;
    block.config = {{"count", U64{9007199254740993ULL}}, {"code", "pass"}};
    block.interfaceConfigs = {
        {.name = "count", .format = {{"type", "uint"}}},
        {.name = "code", .format = {{"type", "python"}}},
        {.name = "console", .format = {{"type", "python-console"}}},
    };
    Runtime::Context::Diagnostic diagnostic;
    diagnostic.status = "ready";
    diagnostic.healthy = true;
    diagnostic.console = {"output"};
    block.metrics.push_back({.name = "python", .format = {{"type", "python-diagnostic"}, {"visibility", "internal"}},
                             .value = diagnostic});
    const auto node = BuildFlowgraphConfigFields("node", block);
    const auto detached = BuildFlowgraphConfigFields("window", block);
    REQUIRE(node.size() == 3);
    REQUIRE(detached.size() == 3);
    U64 count = 0;
    REQUIRE(Parser::Deserialize(node[0].values, node[0].name, count) == Result::SUCCESS);
    REQUIRE(count == U64{9007199254740993ULL});
    REQUIRE(node[0].values == detached[0].values);
    REQUIRE(node[0].format == detached[0].format);
    std::string code;
    REQUIRE(Parser::Deserialize(node[1].values, node[1].name, code) == Result::SUCCESS);
    REQUIRE(code == "pass");
    REQUIRE(node[1].status == "ready");
    REQUIRE(detached[1].consoleOutput == diagnostic.console);
    REQUIRE(detached[1].consoleVisible);
    REQUIRE(node[2].status == "ready");
    REQUIRE(node[2].statusTone == Sakura::StatusTone::Success);
    REQUIRE(node[2].consoleOutput == diagnostic.console);
}

TEST_CASE("Controls report decoding failures rather than silently displaying defaults", "[core][sakura][interface]") {
    FlowgraphConfigFieldInstance field;
    std::vector<std::string> errors;
    FlowgraphConfigFieldConfig config{
        .id = "numeric",
        .name = "amount",
        .format = {{"type", "float"}},
        .values = {{"amount", "invalid"}},
        .onError = [&](Result result, std::string message) {
            REQUIRE(result == Result::ERROR);
            errors.push_back(std::move(message));
        },
    };
    field.update(config);
    REQUIRE(errors.size() == 1);
    REQUIRE(errors.front().find("amount") != std::string::npos);
    for (int i = 0; i < 5; ++i) {
        field.update(config);
    }
    REQUIRE(errors.size() == 1);
    config.values["amount"] = std::string("still invalid");
    field.update(config);
    REQUIRE(errors.size() == 1);
    REQUIRE(field.isSimple());
    config.values["amount"] = 12.0f;
    field.update(config);
    REQUIRE(field.isSimple());
    REQUIRE(errors.size() == 1);
    config.values["amount"] = std::string("invalid again");
    field.update(config);
    REQUIRE(errors.size() == 2);
    field.update(config);
    REQUIRE(errors.size() == 2);
    config.id = "another-field";
    field.update(config);
    REQUIRE(errors.size() == 3);
}

TEST_CASE("Vector controls recover their cached values after a decoding error", "[core][sakura][interface]") {
    FlowgraphConfigVectorField field;
    FlowgraphConfigFieldConfig config{
        .id = "vector",
        .name = "items",
        .format = {{"type", "vector"}, {"value_type", "float"}},
        .values = {{"items", std::vector<F32>{1.0f, 2.0f}}},
    };
    REQUIRE(field.update(config) == Result::SUCCESS);
    auto invalid = config;
    invalid.values["items"] = std::string("invalid");
    REQUIRE(field.update(invalid) == Result::ERROR);
    REQUIRE(field.update(config) == Result::SUCCESS);
    config.format["value_type"] = std::string("uint");
    config.values.clear();
    REQUIRE(field.update(config) == Result::SUCCESS);
}
