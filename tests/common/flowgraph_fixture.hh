#ifndef JETSTREAM_TESTS_FLOWGRAPH_FIXTURE_HH
#define JETSTREAM_TESTS_FLOWGRAPH_FIXTURE_HH

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include "jetstream/flowgraph.hh"

class FlowgraphFixture {
 protected:
    std::unique_ptr<Jetstream::Flowgraph> flowgraph;

 public:
    FlowgraphFixture() {
        flowgraph = std::make_unique<Jetstream::Flowgraph>();
        REQUIRE(flowgraph->create({}, nullptr, nullptr, nullptr) == Jetstream::Result::SUCCESS);
    }

    ~FlowgraphFixture() {
        if (flowgraph) {
            std::vector<std::string> names;
            for (const auto& [name, _] : flowgraph->blockList()) {
                names.push_back(name);
            }
            for (const auto& name : names) {
                flowgraph->blockDestroy(name, false);
            }
            flowgraph->destroy();
        }
    }
};

#endif  // JETSTREAM_TESTS_FLOWGRAPH_FIXTURE_HH
