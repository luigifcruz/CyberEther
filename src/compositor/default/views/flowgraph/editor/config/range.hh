#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH

#include "types.hh"

#include "../field_tooltip.hh"

#include <atomic>
#include <cmath>
#include <limits>
#include <memory>

namespace Jetstream {

struct FlowgraphConfigRangeField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    struct Pending {
        F32 value = 0.0f;
        std::atomic<bool> complete{false};
    };

    void parseFormat();

    Config config;
    std::shared_ptr<Pending> pending;
    Parser::Map parsedFormat;
    std::string unit;
    F32 minValue = 0.0f;
    F32 maxValue = 1.0f;
    F32 value = 0.0f;
    bool unsignedInteger = false;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeRangeInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_RANGE_HH
