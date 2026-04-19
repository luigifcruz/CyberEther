#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Ones Tensor block creates with default config",
                 "[modules][ones_tensor][block]") {
    Blocks::OnesTensor config;
    REQUIRE(flowgraph->blockCreate("ones_default", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("ones_default")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("ones_default")->outputs().count("buffer") == 1);

    const Tensor& out = flowgraph->blockList().at("ones_default")->outputs().at("buffer").tensor;
    REQUIRE(out.shape() == Shape{1});
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("ones_default", saved) == Result::SUCCESS);
    REQUIRE(saved.contains("shape"));
    REQUIRE(saved.contains("dataType"));
}

TEST_CASE_METHOD(FlowgraphFixture, "Ones Tensor block creates a non-default CF32 tensor",
                 "[modules][ones_tensor][block][cf32]") {
    Blocks::OnesTensor config;
    config.shape = {2, 3};
    config.dataType = "CF32";

    REQUIRE(flowgraph->blockCreate("ones_cf32", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("ones_cf32")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("ones_cf32")->outputs().at("buffer").tensor;
    REQUIRE(out.shape() == Shape{2, 3});
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE_THAT(out.at<CF32>(1, 2).real(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(out.at<CF32>(1, 2).imag(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE_METHOD(FlowgraphFixture, "Ones Tensor block recreates on config change",
                 "[modules][ones_tensor][block][reconfigure]") {
    REQUIRE(flowgraph->blockCreate("ones_recfg", "ones_tensor", {}, {}) == Result::SUCCESS);

    Parser::Map update;
    update["shape"] = std::vector<U64>{2, 2};
    update["dataType"] = std::string("CF32");

    REQUIRE(flowgraph->blockReconfigure("ones_recfg", update) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("ones_recfg")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("ones_recfg")->outputs().at("buffer").tensor;
    REQUIRE(out.shape() == Shape{2, 2});
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE_THAT(out.at<CF32>(0, 1).real(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(out.at<CF32>(0, 1).imag(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE_METHOD(FlowgraphFixture, "Ones Tensor block rejects invalid config",
                 "[modules][ones_tensor][block][validation]") {
    Blocks::OnesTensor config;
    config.shape = {0};

    REQUIRE(flowgraph->blockCreate("ones_bad", config, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("ones_bad")->state() == Block::State::Errored);
}
