#ifndef TESTS_PARSER_COMMON_HH
#define TESTS_PARSER_COMMON_HH

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "jetstream/parser.hh"
#include "jetstream/runtime.hh"
#include "jetstream/scheduler.hh"

namespace TestParser {

using namespace Jetstream;

struct InnerConfig {
    U64 gain = 0;
    bool enabled = false;

    JST_SERDES(gain, enabled);
};

struct OuterConfig {
    InnerConfig inner;
    std::string label;

    JST_SERDES(inner, label);
};

struct MapConfig {
    std::unordered_map<std::string, InnerConfig> presets;

    JST_SERDES(presets);
};

struct SequenceConfig {
    std::vector<InnerConfig> steps;

    JST_SERDES(steps);
};

struct OptionalConfig {
    std::optional<std::string> label;
    std::optional<std::vector<U64>> steps;

    JST_SERDES(label, steps);
};

struct PrimitiveVectorConfig {
    std::vector<U64> counts;
    std::vector<F32> ratios;
    std::vector<F64> weights;

    JST_SERDES(counts, ratios, weights);
};

struct NestedVectorConfig {
    std::vector<std::vector<U64>> groups;

    JST_SERDES(groups);
};

struct ThrowingConfig {
    Result serialize(Parser::Map&) const {
        throw std::runtime_error("serialize failure");
    }
};

struct UnsupportedValue {};

inline Parser::Map MakeInnerMap(const U64 gain, const bool enabled) {
    Parser::Map data;
    data["gain"] = gain;
    data["enabled"] = enabled;
    return data;
}

inline Parser::Map MakeStringInnerMap(const U64 gain, const bool enabled) {
    Parser::Map data;
    data["gain"] = std::to_string(gain);
    data["enabled"] = enabled ? std::string("true") : std::string("false");
    return data;
}

}  // namespace TestParser

#endif  // TESTS_PARSER_COMMON_HH
