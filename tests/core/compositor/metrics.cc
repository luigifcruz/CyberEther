#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/metrics.hh"
#include "compositor/default/presenters/flowgraph/editor/node.hh"

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <memory>
#include <ratio>
#include <stdexcept>
#include <string>
#include <utility>

using namespace Jetstream;

namespace {

using Clock = MetricActions::Clock;
using Metric = Flowgraph::View::MetricEntry;

Metric Label(std::string name, std::string value) {
    return {.name = std::move(name), .label = "Value", .format = {{"type", "label"}}, .value = std::move(value)};
}

void Add(DefaultCompositorState& state, const std::string& graph, const std::string& block, Metric metric) {
    auto& blocks = state.flowgraph.blocks[graph];
    for (auto& entry : blocks) {
        if (entry.name == block) {
            entry.data.metrics.push_back(std::move(metric));
            return;
        }
    }
    Flowgraph::View::BlockData data;
    data.metrics.push_back(std::move(metric));
    blocks.push_back({.name = block, .data = std::move(data)});
}

Metric& Current(DefaultCompositorState& state) {
    return state.flowgraph.blocks.at("graph").front().data.metrics.front();
}

void Publish(DefaultCompositorState& state, const std::string& value) {
    state.flowgraph.blocks.clear();
    Add(state, "graph", "soapy", Label("throughput", value));
}

std::string Shown(const DefaultCompositorState& state) {
    return std::any_cast<std::string>(state.flowgraph.blocks.at("graph").front().data.metrics.front().value);
}

struct CopyProbe {
    std::shared_ptr<bool> fail;
    std::string text;

    CopyProbe(std::shared_ptr<bool> fail, std::string text) : fail(std::move(fail)), text(std::move(text)) {}

    CopyProbe(const CopyProbe& other) : fail(other.fail), text(other.text) {
        if (*fail) {
            throw std::runtime_error("Metric copy failed");
        }
    }
};

}  // namespace

TEST_CASE("Block metrics use a sixty-hertz refresh interval", "[core][compositor][metrics]") {
    const auto period = std::chrono::duration<I64, std::ratio<1, 60>>{1};
    CHECK(MetricActions::RefreshInterval >= period);
    CHECK(MetricActions::RefreshInterval - Clock::duration(1) < period);

    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();
    Publish(state, "initial");
    actions.refresh(start);
    Publish(state, "early");
    actions.refresh(start + std::chrono::milliseconds(16));
    CHECK(Shown(state) == "initial");
    Publish(state, "ready");
    actions.refresh(start + std::chrono::milliseconds(17));
    CHECK(Shown(state) == "ready");
}

TEST_CASE("Block metrics refresh at most once per sixty-hertz interval",
          "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();

    Publish(state, "1.0 MB/s");
    actions.refresh(start);
    CHECK(Shown(state) == "1.0 MB/s");

    Publish(state, "1.1 MB/s");
    actions.refresh(start + MetricActions::RefreshInterval / 2);
    CHECK(Shown(state) == "1.0 MB/s");

    Publish(state, "1.15 MB/s");
    actions.refresh(start + MetricActions::RefreshInterval - Clock::duration(1));
    CHECK(Shown(state) == "1.0 MB/s");

    Publish(state, "1.2 MB/s");
    actions.refresh(start + MetricActions::RefreshInterval);
    CHECK(Shown(state) == "1.2 MB/s");

    Publish(state, "1.3 MB/s");
    actions.refresh(start + MetricActions::RefreshInterval + MetricActions::RefreshInterval / 2);
    CHECK(Shown(state) == "1.2 MB/s");
}

TEST_CASE("New and returning block metrics show their current value immediately", "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();

    Publish(state, "1.0 MB/s");
    actions.refresh(start);

    state.flowgraph.blocks.clear();
    actions.refresh(start + MetricActions::RefreshInterval / 4);
    CHECK(actions.entries.empty());

    Publish(state, "2.0 MB/s");
    actions.refresh(start + MetricActions::RefreshInterval / 2);
    CHECK(Shown(state) == "2.0 MB/s");
}

TEST_CASE("Colon-containing metric identities refresh and disappear independently", "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();
    const std::array identities{
        MetricActions::Key{"graph", "a", "b:throughput"},
        MetricActions::Key{"graph", "a:b", "throughput"},
        MetricActions::Key{"graph:a", "b", "throughput"},
        MetricActions::Key{"graph", "a", "throughput"},
    };
    const auto publish = [&](const std::string& suffix, bool missing = false) {
        state.flowgraph.blocks.clear();
        for (std::size_t i = 0; i < identities.size(); ++i) {
            if (missing && i == 0) {
                continue;
            }
            const auto& [graph, block, name] = identities[i];
            Add(state, graph, block, Label(name, std::to_string(i) + suffix));
        }
    };
    const auto check = [&](const std::string& suffix, bool missing = false) {
        CHECK(actions.entries.size() == identities.size() - (missing ? 1 : 0));
        for (const auto& [graph, blocks] : state.flowgraph.blocks) {
            for (const auto& block : blocks) {
                for (const auto& metric : block.data.metrics) {
                    const MetricActions::Key key{graph, block.name, metric.name};
                    const auto it = std::find(identities.begin(), identities.end(), key);
                    REQUIRE(it != identities.end());
                    CHECK(std::any_cast<std::string>(metric.value) == std::to_string(it - identities.begin()) + suffix);
                }
            }
        }
    };

    publish(" initial");
    actions.refresh(start);
    check(" initial");
    publish(" ignored");
    actions.refresh(start + MetricActions::RefreshInterval / 2);
    check(" initial");
    publish(" next");
    actions.refresh(start + MetricActions::RefreshInterval);
    check(" next");
    publish(" ignored", true);
    actions.refresh(start + MetricActions::RefreshInterval + Clock::duration(1));
    check(" next", true);
    CHECK_FALSE(actions.entries.contains(identities[0]));

    publish(" returning");
    actions.refresh(start + MetricActions::RefreshInterval + Clock::duration(2));
    CHECK(actions.entries.size() == identities.size());
    CHECK(std::any_cast<std::string>(state.flowgraph.blocks.at("graph")[0].data.metrics[0].value) == "0 returning");
    CHECK(std::any_cast<std::string>(state.flowgraph.blocks.at("graph")[0].data.metrics[1].value) == "3 next");
    CHECK(std::any_cast<std::string>(state.flowgraph.blocks.at("graph")[1].data.metrics[0].value) == "1 next");
    CHECK(std::any_cast<std::string>(state.flowgraph.blocks.at("graph:a")[0].data.metrics[0].value) == "2 next");
    publish(" latest");
    actions.refresh(start + 2 * MetricActions::RefreshInterval + Clock::duration(2));
    check(" latest");

    state.flowgraph.blocks.at("graph").erase(state.flowgraph.blocks.at("graph").begin() + 1);
    state.flowgraph.blocks.erase("graph:a");
    actions.refresh(start + 2 * MetricActions::RefreshInterval + Clock::duration(3));
    CHECK(actions.entries.size() == 2);
    CHECK_FALSE(actions.entries.contains(identities[1]));
    CHECK_FALSE(actions.entries.contains(identities[2]));
    state.flowgraph.blocks.clear();
    actions.refresh(start + 2 * MetricActions::RefreshInterval + Clock::duration(4));
    CHECK(actions.entries.empty());
    actions.refresh(start + 2 * MetricActions::RefreshInterval + Clock::duration(5));
    CHECK(actions.entries.empty());
}

TEST_CASE("Metric schema changes reach the presenter with a compatible value immediately", "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();

    Publish(state, "old string");
    actions.refresh(start);
    Current(state) = {
        .name = "throughput", .label = "Progress", .format = {{"type", "progressbar"}}, .help = "Current progress",
        .value = std::pair<std::string, F32>{"50%", 0.5f},
    };
    const auto changed = start + Clock::duration(1);
    actions.refresh(changed);
    FlowgraphNode::BlockData shown;
    shown.metrics = BuildFlowgraphMetrics("graph:soapy", state.flowgraph.blocks.at("graph").front().data);
    REQUIRE(shown.metrics.size() == 1);
    CHECK(Parser::Get<std::string>(shown.metrics[0].format, "type") == "progressbar");
    CHECK(shown.metrics[0].label == "Progress");
    CHECK(shown.metrics[0].help == "Current progress");
    const auto* value = std::any_cast<std::pair<std::string, F32>>(&shown.metrics[0].value);
    REQUIRE(value != nullptr);
    CHECK(value->first == "50%");
    CHECK(value->second == 0.5f);

    Current(state).value = std::pair<std::string, F32>{"75%", 0.75f};
    actions.refresh(changed + MetricActions::RefreshInterval - Clock::duration(1));
    CHECK(std::any_cast<std::pair<std::string, F32>>(Current(state).value).second == 0.5f);
    Current(state).value = std::pair<std::string, F32>{"80%", 0.8f};
    actions.refresh(changed + MetricActions::RefreshInterval);
    CHECK(std::any_cast<std::pair<std::string, F32>>(Current(state).value).second == 0.8f);

    Current(state) = Label("throughput", "returned label");
    actions.refresh(changed + MetricActions::RefreshInterval + Clock::duration(1));
    shown.metrics = BuildFlowgraphMetrics("graph:soapy", state.flowgraph.blocks.at("graph").front().data);
    REQUIRE(shown.metrics.size() == 1);
    CHECK(Parser::Get<std::string>(shown.metrics[0].format, "type") == "label");
    CHECK(std::any_cast<std::string>(shown.metrics[0].value) == "returned label");
}

TEST_CASE("Metric metadata changes refresh values without bypassing the next interval", "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    auto now = Clock::now();
    Publish(state, "initial");
    Current(state).format["unit"] = std::string("bytes/s");
    Current(state).format["options"] = Parser::Map{{"precision", U64{2}}};
    actions.refresh(now);

    const auto check = [&] {
        now += MetricActions::RefreshInterval / 4;
        Current(state).value = std::string("changed");
        actions.refresh(now);
        CHECK(Shown(state) == "changed");
        Current(state).value = std::string("ignored");
        actions.refresh(now + MetricActions::RefreshInterval - Clock::duration(1));
        CHECK(Shown(state) == "changed");
        Current(state).value = std::string("boundary");
        now += MetricActions::RefreshInterval;
        actions.refresh(now);
        CHECK(Shown(state) == "boundary");
    };

    SECTION("Label") {
        Current(state).label = "New label";
    }
    SECTION("Help") {
        Current(state).help = "New help";
    }
    SECTION("Unit") {
        Current(state).format["unit"] = std::string("KiB/s");
    }
    SECTION("Format type") {
        Current(state).format["type"] = std::string("progressbar");
    }
    SECTION("Nested format") {
        std::any_cast<Parser::Map&>(Current(state).format["options"])["precision"] = U64{3};
    }
    SECTION("Removed format") {
        Current(state).format.erase("unit");
    }
    check();
}

TEST_CASE("Metric payload type and empty value transitions refresh immediately", "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();
    Publish(state, "label");
    actions.refresh(start);

    Current(state).value = U64{42};
    actions.refresh(start + Clock::duration(1));
    CHECK(std::any_cast<U64>(Current(state).value) == 42);
    Current(state).value = U64{43};
    actions.refresh(start + Clock::duration(2));
    CHECK(std::any_cast<U64>(Current(state).value) == 42);
    Current(state).value.reset();
    actions.refresh(start + Clock::duration(3));
    CHECK_FALSE(Current(state).value.has_value());
    Current(state).value = std::string("ready");
    actions.refresh(start + Clock::duration(4));
    CHECK(Shown(state) == "ready");
}

TEST_CASE("Metric cache owns snapshots and releases missing payloads", "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();
    Publish(state, "first");
    actions.refresh(start);
    std::any_cast<std::string&>(Current(state).value) = "mutated";
    actions.refresh(start + Clock::duration(1));
    CHECK(Shown(state) == "first");

    auto payload = std::make_shared<int>(42);
    const std::weak_ptr<int> lifetime = payload;
    Current(state).value = payload;
    actions.refresh(start + Clock::duration(2));
    payload.reset();
    state.flowgraph.blocks.clear();
    CHECK_FALSE(lifetime.expired());
    actions.refresh(start + Clock::duration(3));
    CHECK(lifetime.expired());
}

TEST_CASE("Failed metric copies do not leave partial snapshots or advance the interval", "[core][compositor][metrics]") {
    DefaultCompositorState state;
    MetricActions actions(state.flowgraph);
    const auto start = Clock::now();
    auto fail = std::make_shared<bool>(false);
    Publish(state, "initial");
    Current(state).value = CopyProbe{fail, "first"};
    *fail = true;
    CHECK_THROWS_AS(actions.refresh(start), std::runtime_error);
    CHECK(actions.entries.empty());
    *fail = false;
    actions.refresh(start + Clock::duration(1));
    CHECK(std::any_cast<const CopyProbe&>(Current(state).value).text == "first");

    Current(state).value = CopyProbe{fail, "next"};
    *fail = true;
    CHECK_THROWS_AS(actions.refresh(start + MetricActions::RefreshInterval + Clock::duration(1)), std::runtime_error);
    *fail = false;
    actions.refresh(start + MetricActions::RefreshInterval + Clock::duration(2));
    CHECK(std::any_cast<const CopyProbe&>(Current(state).value).text == "next");
    Current(state).value = CopyProbe{fail, "ignored"};
    actions.refresh(start + MetricActions::RefreshInterval + Clock::duration(3));
    CHECK(std::any_cast<const CopyProbe&>(Current(state).value).text == "next");
}
