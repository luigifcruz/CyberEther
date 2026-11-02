#include "base.hh"

#include <utility>

namespace Jetstream::Agent::Detail {

std::string Wire::Field(const nlohmann::json& object, const char* name) {
    const auto field = object.find(name);
    if (field == object.end()) {
        return {};
    }
    if (field->is_string()) {
        return field->get<std::string>();
    }
    return field->is_number_integer() ? field->dump() : std::string{};
}

std::string Wire::Error(const nlohmann::json& payload) {
    if (!payload.is_object()) {
        return {};
    }
    const auto found = payload.find("error");
    const auto& value = found != payload.end() ? *found : payload;
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return value.is_object() ? Field(value, "message") : std::string{};
}

nlohmann::json Wire::Replay(const Provider::Message& message, const std::string& host,
                            const std::string& path, const std::string& model) {
    const auto& saved = message.state;
    if (message.role == "assistant" && saved && saved->host == host &&
        saved->path == path && saved->model == model) {
        return nlohmann::json::parse(saved->payload);
    }
    return {};
}

bool Wire::Empty(const Provider::Message& message, const nlohmann::json& state) {
    return message.role == "assistant" && message.content.empty() && message.toolCalls.empty() &&
           message.images.empty() && !(state.is_object() && !state.empty());
}

nlohmann::json Wire::Function(const Provider::Function& tool) {
    auto parameters = nlohmann::json::parse(tool.parametersJson, nullptr, false);
    if (!parameters.is_object()) {
        parameters = {{"type", "object"}, {"properties", nlohmann::json::object()}};
    }
    return {
        {"name", tool.name},
        {"description", tool.description},
        {"parameters", std::move(parameters)},
    };
}

nlohmann::json Wire::FunctionCall(const Provider::FunctionCall& call) {
    return {
        {"id", call.id},
        {"type", "function"},
        {"function", {{"name", call.name}, {"arguments", call.arguments}}},
    };
}

}  // namespace Jetstream::Agent::Detail
