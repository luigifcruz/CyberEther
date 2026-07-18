#ifndef TESTS_CORE_SERIALIZATION_FIXTURE_HH
#define TESTS_CORE_SERIALIZATION_FIXTURE_HH

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "jetstream/parser.hh"
#include "jetstream/runtime.hh"
#include "jetstream/scheduler.hh"

namespace TestSerialization {

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
    std::vector<std::string> names;

    JST_SERDES(counts, ratios, weights, names);
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

struct FailingSerializeConfig {
    Result serialize(Parser::Map&) const {
        return Result::ERROR;
    }
};

struct ThrowingDeserializeConfig {
    Result deserialize(const Parser::Map&) {
        throw std::runtime_error("deserialize failure");
    }
};

struct ResultThrowingDeserializeConfig {
    Result deserialize(const Parser::Map&) {
        throw Result::FATAL;
    }
};

struct UnknownThrowingDeserializeConfig {
    Result deserialize(const Parser::Map&) {
        throw 42;
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

}  // namespace TestSerialization

#endif  // TESTS_CORE_SERIALIZATION_FIXTURE_HH
