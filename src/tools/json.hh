#ifndef JETSTREAM_TOOLS_JSON_HH
#define JETSTREAM_TOOLS_JSON_HH

#include "jetstream/parser.hh"

#include <any>

#include <nlohmann/json.hpp>

namespace Jetstream::Value {

JETSTREAM_API Result ToJson(const Parser::Map& data, nlohmann::json& json);
JETSTREAM_API Result ToJson(const std::any& value, nlohmann::json& json);
JETSTREAM_API bool TryToJson(const std::any& value, nlohmann::json& json);
JETSTREAM_API Result FromJson(const nlohmann::json& json, Parser::Map& data);

}  // namespace Jetstream::Value

#endif  // JETSTREAM_TOOLS_JSON_HH
