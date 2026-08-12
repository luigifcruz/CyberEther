#ifndef JETSTREAM_TESTS_SUPPORT_FLOWGRAPH_FIXTURE_HH
#define JETSTREAM_TESTS_SUPPORT_FLOWGRAPH_FIXTURE_HH

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include "jetstream/flowgraph_view.hh"

#include "synthetic_graph.hh"

inline Jetstream::Flowgraph::View::BlockData ViewBlock(Jetstream::Flowgraph& flowgraph,
                                                        const std::string& name) {
    Jetstream::Flowgraph::View::BlockData data;
    REQUIRE(flowgraph.view().block(name, data) == Jetstream::Result::SUCCESS);
    return data;
}

class FlowgraphFixture {
 protected:
    std::unique_ptr<Jetstream::Flowgraph> flowgraph;

    Jetstream::Flowgraph::View::BlockData viewBlock(const std::string& name) {
        return ViewBlock(*flowgraph, name);
    }

 public:
    FlowgraphFixture() {
        REQUIRE(TestFlowgraph::RegisterSyntheticGraph() == Jetstream::Result::SUCCESS);
        TestFlowgraph::syntheticFaultState().reset();
        flowgraph = std::make_unique<Jetstream::Flowgraph>();
        REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) ==
                Jetstream::Result::SUCCESS);
    }

    ~FlowgraphFixture() {
        TestFlowgraph::syntheticFaultState().reset();
        if (!flowgraph) {
            return;
        }

        std::vector<std::string> names;
        const auto keysResult = flowgraph->view().keys(names);
        CHECK(keysResult == Jetstream::Result::SUCCESS);
        if (keysResult == Jetstream::Result::SUCCESS) {
            for (const auto& name : names) {
                CHECK(flowgraph->blockDestroy(name, false) ==
                      Jetstream::Result::SUCCESS);
            }
        }
        CHECK(flowgraph->destroy() == Jetstream::Result::SUCCESS);
    }
};

#endif  // JETSTREAM_TESTS_SUPPORT_FLOWGRAPH_FIXTURE_HH
