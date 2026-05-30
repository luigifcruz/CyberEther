#ifndef JETSTREAM_TESTS_FLOWGRAPH_FIXTURE_HH
#define JETSTREAM_TESTS_FLOWGRAPH_FIXTURE_HH

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"

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
        flowgraph = std::make_unique<Jetstream::Flowgraph>();
        REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) == Jetstream::Result::SUCCESS);
    }

    ~FlowgraphFixture() {
        if (flowgraph) {
            std::vector<std::string> names;
            REQUIRE(flowgraph->view().keys(names) == Jetstream::Result::SUCCESS);
            for (const auto& name : names) {
                flowgraph->blockDestroy(name, false);
            }
            flowgraph->destroy();
        }
    }
};

#endif  // JETSTREAM_TESTS_FLOWGRAPH_FIXTURE_HH
