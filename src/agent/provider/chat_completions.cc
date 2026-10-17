#include "support/base.hh"

#include <algorithm>
#include <iterator>
#include <map>
#include <string_view>
#include <utility>

namespace Jetstream::Agent::Detail {

namespace {

using Config = Provider::Config;
using Request = Provider::Request;
using Kind = Provider::Error::Kind;

nlohmann::json EncodeContent(const Provider::Message& message) {
    if (message.images.empty()) {
        return message.content;
    }

    auto parts = nlohmann::json::array();
    if (!message.content.empty()) {
        parts.push_back({
            {"type", "text"},
            {"text", message.content},
        });
    }
    for (const auto& image : message.images) {
        parts.push_back({
            {"type", "image_url"},
            {"image_url", {{"url", image}}},
        });
    }
    return parts;
}

nlohmann::json EncodeMessage(const Provider::Message& message, const nlohmann::json& state) {
    nlohmann::json item = {{"role", message.role}};
    item["content"] = EncodeContent(message);

    if (!message.toolCallId.empty()) {
        item["tool_call_id"] = message.toolCallId;
    }
    if (!message.toolCalls.empty()) {
        auto& calls = item["tool_calls"] = nlohmann::json::array();
        for (const auto& call : message.toolCalls) {
            calls.push_back(Wire::FunctionCall(call));
        }
    }

    if (state.is_object()) {
        item.update(state);
    }
    return item;
}

bool MarkCacheBreakpoint(nlohmann::json& message) {
    const nlohmann::json breakpoint = {{"type", "ephemeral"}};
    auto& content = message["content"];

    if (content.is_string()) {
        if (content.get_ref<const std::string&>().empty()) {
            return false;
        }

        auto text = content.get<std::string>();
        content = nlohmann::json::array({{
            {"type", "text"},
            {"text", std::move(text)},
            {"cache_control", breakpoint},
        }});
        return true;
    }

    if (content.is_array()) {
        for (auto it = content.rbegin(); it != content.rend(); ++it) {
            if (it->is_object() && it->value("type", "") == "text") {
                (*it)["cache_control"] = breakpoint;
                return true;
            }
        }
    }

    return false;
}

bool UsesCacheBreakpoints(const Config& config, const Request& request) {
    const auto& models = Provider::Profile::Find(config.kind).cacheBreakpointModels;
    return std::any_of(models.begin(), models.end(), [&](const std::string& prefix) {
        return request.model.starts_with(prefix);
    });
}

void MarkCacheBreakpoints(nlohmann::json& messages) {
    if (messages.empty()) {
        return;
    }

    if (messages.front()["role"] == "system") {
        MarkCacheBreakpoint(messages.front());
    }

    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (MarkCacheBreakpoint(*it)) {
            break;
        }
    }
}

nlohmann::json EncodeMessages(const Config& config, const Request& request) {
    auto encoded = nlohmann::json::array();
    for (const auto& message : request.messages) {
        const auto state = Wire::Replay(message, config.host, config.path, request.model);
        if (Wire::Empty(message, state)) {
            continue;
        }
        encoded.push_back(EncodeMessage(message, state));
    }

    if (UsesCacheBreakpoints(config, request)) {
        MarkCacheBreakpoints(encoded);
    }
    return encoded;
}

nlohmann::json EncodeTools(const std::span<const Provider::Function> tools) {
    auto functions = nlohmann::json::array();
    for (const auto& tool : tools) {
        functions.push_back({
            {"type", "function"},
            {"function", Wire::Function(tool)},
        });
    }
    return functions;
}

struct DetailIdentity {
    const nlohmann::json* index = nullptr;
    std::string id;
    std::string type;

    bool anonymous() const {
        return !index && id.empty();
    }
};

DetailIdentity Identify(const nlohmann::json& detail) {
    const auto index = detail.find("index");
    const bool indexed = index != detail.end() && index->is_number_integer();

    DetailIdentity identity;
    identity.index = indexed ? &*index : nullptr;
    identity.id = Wire::Field(detail, "id");
    identity.type = Wire::Field(detail, "type");
    return identity;
}

bool SameDetail(const nlohmann::json& candidate, const DetailIdentity& identity) {
    const auto candidateType = Wire::Field(candidate, "type");
    const auto candidateId = Wire::Field(candidate, "id");

    const bool typeKnown = !identity.type.empty() && !candidateType.empty();
    if (typeKnown && candidateType != identity.type) {
        return false;
    }

    if (identity.index) {
        if (!candidate.contains("index") || candidate["index"] != *identity.index) {
            return false;
        }
        return identity.id.empty() || candidateId.empty() || candidateId == identity.id;
    }

    if (!identity.id.empty()) {
        return candidateId == identity.id;
    }
    return !candidate.contains("index") && candidateId.empty();
}

void MergeDetail(nlohmann::json& merged, const nlohmann::json& detail) {
    for (const auto& [key, value] : detail.items()) {
        if (value.is_null()) {
            continue;
        }

        const bool streamed = key == "text" || key == "summary" || key == "data" || key == "signature";
        const bool appendable = streamed && value.is_string() && merged.contains(key) && merged[key].is_string();
        if (appendable) {
            merged[key] = merged[key].get<std::string>() + value.get<std::string>();
        } else {
            merged[key] = value;
        }
    }
}

void AccumulateToolCall(Provider::FunctionCall& accumulated, const nlohmann::json& call) {
    if (call.contains("id") && call["id"].is_string()) {
        accumulated.id = call["id"].get<std::string>();
    }

    const auto function = call.find("function");
    if (function == call.end() || !function->is_object()) {
        return;
    }

    if (function->contains("name") && (*function)["name"].is_string()) {
        accumulated.name = (*function)["name"].get<std::string>();
    }
    if (function->contains("arguments") && (*function)["arguments"].is_string()) {
        accumulated.arguments += (*function)["arguments"].get<std::string>();
    }
}

class CompletionStream final : public EventStream {
 public:
    std::string payload() const override {
        nlohmann::json fields = nlohmann::json::object();
        if (!reasoningDetails.empty()) {
            fields["reasoning_details"] = reasoningDetails;
        } else if (!reasoningText.empty()) {
            fields[reasoningKey] = reasoningText;
        }

        if (fields.empty()) {
            return "";
        }
        return fields.dump();
    }

 protected:
    bool decode(const nlohmann::json& chunk) override {
        readUsage(chunk, ChatCompletionsUsage);

        const auto choices = chunk.find("choices");
        if (choices == chunk.end()) {
            return fail(Kind::Protocol, "The provider response is missing the 'choices' array.");
        }
        if (!choices->is_array()) {
            return fail(Kind::Protocol, "The provider response has an invalid 'choices' field.");
        }
        if (choices->empty()) {
            return true;
        }

        return decodeChoice(choices->front());
    }

    bool complete() override {
        for (auto& [index, call] : calls) {
            if (call.id.empty() || call.name.empty()) {
                return fail(Kind::Protocol, "The provider returned an incomplete tool call.");
            }
            toolCalls.push_back(std::move(call));
        }
        return true;
    }

 private:
    bool decodeChoice(const nlohmann::json& choice) {
        if (!choice.is_object()) {
            return fail(Kind::Protocol, "The provider returned an invalid completion choice.");
        }
        if (choice.contains("error")) {
            error = Diagnostics::Payload(choice);
            return false;
        }

        sawChoice = true;
        if (!readFinishReason(choice)) {
            return false;
        }

        const auto delta = choice.find("delta");
        if (delta == choice.end() || !delta->is_object()) {
            if (finishedChoice) {
                return true;
            }
            return fail(Kind::Protocol, "The provider response is missing the completion delta.");
        }

        return decodeDelta(*delta);
    }

    bool readFinishReason(const nlohmann::json& choice) {
        const auto finish = choice.find("finish_reason");
        if (finish == choice.end() || !finish->is_string()) {
            return true;
        }

        finishedChoice = true;
        truncated = *finish == "length";

        if (*finish == "error" || *finish == "content_filter") {
            return fail(Kind::InvalidRequest,
                        "The provider stopped the response: " + finish->get<std::string>());
        }
        return true;
    }

    bool decodeDelta(const nlohmann::json& delta) {
        if (!readReasoningDetails(delta)) {
            readReasoningText(delta);
        }

        if (!readToolCalls(delta)) {
            return false;
        }

        const auto content = delta.find("content");
        if (content != delta.end() && content->is_string()) {
            token(content->get<std::string>());
        }
        return true;
    }

    bool readReasoningDetails(const nlohmann::json& delta) {
        const auto details = delta.find("reasoning_details");
        if (details == delta.end() || !details->is_array()) {
            return false;
        }

        bool detailed = false;
        size_t anonymousPosition = 0;
        for (const auto& detail : *details) {
            if (!detail.is_object()) {
                continue;
            }

            outputStarted = true;
            captureReasoningDetail(detail, anonymousPosition);
            detailed |= emitReasoningDetail(detail);
        }
        return detailed;
    }

    bool emitReasoningDetail(const nlohmann::json& detail) {
        for (const char* key : {"summary", "text"}) {
            const auto value = detail.find(key);
            const bool present = value != detail.end() && value->is_string() &&
                                 !value->get_ref<const std::string&>().empty();
            if (!present) {
                continue;
            }

            const auto index = detail.find("index");
            const bool summaryType = detail.contains("type") && detail["type"] == "reasoning.summary";
            const bool summary = std::string_view(key) == "summary" || summaryType;

            I64 position = lastReasoningIndex;
            if (index != detail.end() && index->is_number_integer()) {
                position = index->get<I64>();
            }

            reasoning(value->get<std::string>(), position, summary);
            return true;
        }
        return false;
    }

    void readReasoningText(const nlohmann::json& delta) {
        for (const char* key : {"reasoning", "reasoning_content"}) {
            const auto value = delta.find(key);
            const bool present = value != delta.end() && value->is_string() &&
                                 !value->get_ref<const std::string&>().empty();
            if (!present) {
                continue;
            }

            outputStarted = true;
            reasoning(value->get<std::string>(), lastReasoningIndex, false);

            if (reasoningKey.empty()) {
                reasoningKey = key;
            }
            if (reasoningKey == key) {
                reasoningText += value->get<std::string>();
            }
        }
    }

    bool readToolCalls(const nlohmann::json& delta) {
        const auto toolDeltas = delta.find("tool_calls");
        if (toolDeltas == delta.end() || toolDeltas->is_null()) {
            return true;
        }
        if (!toolDeltas->is_array()) {
            return fail(Kind::Protocol, "The provider returned invalid tool calls.");
        }

        for (const auto& call : *toolDeltas) {
            outputStarted = true;

            const bool indexed = call.is_object() && call.contains("index") && call["index"].is_number_integer();
            if (!indexed) {
                return fail(Kind::Protocol, "A streamed tool call is missing its index.");
            }

            auto& accumulated = calls[call["index"].get<I64>()];
            const bool named = !accumulated.name.empty();
            AccumulateToolCall(accumulated, call);
            if (!named && !accumulated.name.empty()) {
                announce(accumulated.name);
            }
        }
        return true;
    }

    void captureReasoningDetail(const nlohmann::json& detail, size_t& anonymousPosition) {
        const auto identity = Identify(detail);
        const bool anonymous = identity.anonymous();
        const size_t start = anonymous ? anonymousPosition : 0;

        auto found = std::find_if(reasoningDetails.begin() + start, reasoningDetails.end(),
                                  [&](const nlohmann::json& candidate) {
            return SameDetail(candidate, identity);
        });
        if (found == reasoningDetails.end()) {
            reasoningDetails.push_back(nlohmann::json::object());
            found = std::prev(reasoningDetails.end());
        }

        if (anonymous) {
            anonymousPosition = std::distance(reasoningDetails.begin(), found) + 1;
        }

        MergeDetail(*found, detail);
    }

    std::map<I64, Provider::FunctionCall> calls;
    std::vector<nlohmann::json> reasoningDetails;
    std::string reasoningText;
    std::string reasoningKey;
};

}  // namespace

nlohmann::json ChatCompletions::Encode(const Config& config, const Request& request, bool reasoning) {
    nlohmann::json body = {
        {"model", request.model},
        {"stream", true},
    };
    body["stream_options"] = {{"include_usage", true}};

    if (reasoning && config.requestReasoning) {
        body["reasoning"] = {{"enabled", true}};
    }

    if (!request.reasoningEffort.empty()) {
        if (Provider::Profile::Find(config.kind).nestedReasoningEffort) {
            body["reasoning"]["effort"] = request.reasoningEffort;
        } else {
            body["reasoning_effort"] = request.reasoningEffort;
        }
    }

    body["messages"] = EncodeMessages(config, request);

    if (!request.tools.empty()) {
        body["tools"] = EncodeTools(request.tools);
        if (!request.allowTools) {
            body["tool_choice"] = "none";
        }
    }

    if (request.maxOutputTokens) {
        body["max_tokens"] = request.maxOutputTokens;
    }

    return body;
}

std::unique_ptr<EventStream> ChatCompletions::Decoder() {
    return std::make_unique<CompletionStream>();
}

}  // namespace Jetstream::Agent::Detail
