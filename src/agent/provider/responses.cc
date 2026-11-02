#include "support/base.hh"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>

namespace Jetstream::Agent::Detail {

namespace {

using Config = Provider::Config;
using Request = Provider::Request;
using Kind = Provider::Error::Kind;

bool AppendResponseHistory(nlohmann::json& input, const nlohmann::json& state) {
    const auto unresolved = [](const nlohmann::json& item) {
        return Wire::Field(item, "type") == "reasoning" &&
               Wire::Field(item, "encrypted_content").empty();
    };
    const bool omittedReasoning = std::any_of(state.begin(), state.end(), unresolved);
    const auto start = input.size();

    for (auto item : state) {
        if (unresolved(item)) {
            continue;
        }

        const auto type = Wire::Field(item, "type");
        const bool replayable = type == "message" || type == "function_call";
        if (omittedReasoning && replayable) {
            item.erase("id");
        }

        item.erase("created_by");
        input.push_back(std::move(item));
    }

    return input.size() != start;
}

void AppendMessage(nlohmann::json& input, const Provider::Message& message) {
    if (message.role == "tool") {
        input.push_back(Responses::FunctionOutput(message.toolCallId, message.content));
        return;
    }

    nlohmann::json item = {{"role", message.role}};
    if (message.images.empty()) {
        item["content"] = message.content;
    } else {
        item["content"] = Responses::InputContent(message.content, message.images);
    }

    if (message.toolCalls.empty()) {
        input.push_back(std::move(item));
        return;
    }
    if (!message.content.empty()) {
        input.push_back(std::move(item));
    }

    for (const auto& call : message.toolCalls) {
        input.push_back({
            {"type", "function_call"},
            {"call_id", call.id},
            {"name", call.name},
            {"arguments", call.arguments},
        });
    }
}

nlohmann::json EncodeInput(const Config& config, const Request& request) {
    auto input = nlohmann::json::array();
    for (const auto& message : request.messages) {
        const auto state = Wire::Replay(message, config.host, config.path, request.model);
        if (state.is_array() && AppendResponseHistory(input, state)) {
            continue;
        }

        if (Wire::Empty(message, state)) {
            continue;
        }
        AppendMessage(input, message);
    }
    return input;
}

nlohmann::json EncodeTools(const std::span<const Provider::Function> tools) {
    auto functions = nlohmann::json::array();
    for (const auto& tool : tools) {
        functions.push_back(Responses::Function(tool));
    }
    return functions;
}

std::string IncompleteReason(const nlohmann::json& chunk) {
    const auto response = chunk.find("response");
    if (response == chunk.end() || !response->is_object()) {
        return {};
    }

    const auto details = response->find("incomplete_details");
    if (details == response->end() || !details->is_object()) {
        return {};
    }
    return Wire::Field(*details, "reason");
}

class ResponseStream final : public EventStream {
 public:
    std::string payload() const override {
        auto output = nlohmann::json::array();
        for (const auto& [index, item] : items) {
            output.push_back(item);
        }

        if (output.empty()) {
            return "";
        }
        return output.dump();
    }

 protected:
    bool decode(const nlohmann::json& chunk) override {
        using Handler = bool (ResponseStream::*)(const nlohmann::json&);
        static const std::unordered_map<std::string, Handler> handlers{
            {"response.output_text.delta", &ResponseStream::textDelta},
            {"response.refusal.delta", &ResponseStream::textDelta},
            {"response.reasoning_summary_text.delta", &ResponseStream::summaryDelta},
            {"response.function_call_arguments.delta", &ResponseStream::argumentsDelta},
            {"response.output_item.added", &ResponseStream::outputItemAdded},
            {"response.output_item.done", &ResponseStream::outputItemDone},
            {"response.completed", &ResponseStream::completed},
            {"response.failed", &ResponseStream::failed},
            {"response.incomplete", &ResponseStream::failed},
        };

        const auto handler = handlers.find(Wire::Field(chunk, "type"));
        if (handler == handlers.end()) {
            return true;
        }
        return (this->*handler->second)(chunk);
    }

    bool complete() override {
        if (!collectToolCalls()) {
            return false;
        }

        if (text.empty()) {
            snapshotText();
        }
        return true;
    }

 private:
    bool textDelta(const nlohmann::json& chunk) {
        token(Wire::Field(chunk, "delta"));
        return true;
    }

    bool summaryDelta(const nlohmann::json& chunk) {
        outputStarted |= !Wire::Field(chunk, "delta").empty();
        reasoning(Wire::Field(chunk, "delta"), 0, true);
        return true;
    }

    bool argumentsDelta(const nlohmann::json& chunk) {
        outputStarted |= !Wire::Field(chunk, "delta").empty();
        return true;
    }

    bool outputItemAdded(const nlohmann::json& chunk) {
        const auto item = chunk.find("item");
        const bool functionCall = item != chunk.end() && item->is_object() &&
                                  Wire::Field(*item, "type") == "function_call";
        if (functionCall) {
            announce(Wire::Field(*item, "name"));
        }
        return true;
    }

    bool outputItemDone(const nlohmann::json& chunk) {
        const auto item = chunk.find("item");
        if (item == chunk.end() || !item->is_object()) {
            return true;
        }

        outputStarted = true;

        I64 position = 0;
        const auto index = chunk.find("output_index");
        if (index != chunk.end() && index->is_number_integer()) {
            position = index->get<I64>();
        } else if (!items.empty()) {
            position = items.rbegin()->first + 1;
        }

        items[position] = *item;
        return true;
    }

    bool completed(const nlohmann::json& chunk) {
        finishedChoice = true;
        sawChoice = true;

        const auto response = chunk.find("response");
        if (response == chunk.end() || !response->is_object()) {
            return true;
        }

        readUsage(*response, ResponsesUsage);

        const auto output = response->find("output");
        if (output != response->end() && output->is_array() && !output->empty()) {
            items.clear();
            for (size_t i = 0; i < output->size(); ++i) {
                items[i] = (*output)[i];
            }
        }

        return true;
    }

    bool failed(const nlohmann::json& chunk) {
        if (chunk.contains("response") && chunk["response"].is_object()) {
            readUsage(chunk["response"], ResponsesUsage);
        }

        if (const auto* response = Responses::Failed(chunk)) {
            error = Diagnostics::Payload(*response);
            return false;
        }

        if (IncompleteReason(chunk) == "max_output_tokens") {
            return fail(Kind::OutputLimit, Truncated);
        }

        if (Wire::Field(chunk, "type") == "response.failed") {
            return fail(Kind::InvalidRequest, "The provider failed to complete the response.");
        }
        return fail(Kind::InvalidRequest, "The provider returned an incomplete response.");
    }

    bool collectToolCalls() {
        for (const auto& [index, item] : items) {
            if (!item.is_object()) {
                return fail(Kind::Protocol, "The provider returned an invalid response output item.");
            }

            auto call = Responses::FunctionCall(item);
            if (!call) {
                continue;
            }
            if (call->id.empty() || call->name.empty()) {
                return fail(Kind::Protocol, "The provider returned an incomplete tool call.");
            }
            toolCalls.push_back(std::move(*call));
        }
        return true;
    }

    void snapshotText() {
        for (const auto& [index, item] : items) {
            if (Wire::Field(item, "type") != "message") {
                continue;
            }

            const auto content = item.find("content");
            if (content == item.end() || !content->is_array()) {
                continue;
            }

            for (const auto& part : *content) {
                const char* key = "text";
                if (Wire::Field(part, "type") == "refusal") {
                    key = "refusal";
                }
                text += Wire::Field(part, key);
            }
        }

        if (!text.empty() && onToken) {
            onToken(text);
        }
    }

    std::map<I64, nlohmann::json> items;
};

}  // namespace

nlohmann::json Responses::Function(const Provider::Function& tool) {
    auto function = Wire::Function(tool);
    function["type"] = "function";
    function["strict"] = false;
    return function;
}

nlohmann::json Responses::FunctionOutput(const std::string& callId, const std::string& output) {
    return {
        {"type", "function_call_output"},
        {"call_id", callId},
        {"output", output},
    };
}

nlohmann::json Responses::InputContent(const std::string& text, const std::vector<std::string>& images) {
    auto parts = nlohmann::json::array();
    if (!text.empty()) {
        parts.push_back({
            {"type", "input_text"},
            {"text", text},
        });
    }
    for (const auto& image : images) {
        parts.push_back({
            {"type", "input_image"},
            {"image_url", image},
        });
    }
    return parts;
}

std::optional<Provider::FunctionCall> Responses::FunctionCall(const nlohmann::json& item) {
    if (!item.is_object() || Wire::Field(item, "type") != "function_call") {
        return std::nullopt;
    }

    Provider::FunctionCall call{
        Wire::Field(item, "call_id"),
        Wire::Field(item, "name"),
        Wire::Field(item, "arguments"),
    };
    if (call.arguments.empty()) {
        call.arguments = "{}";
    }
    return call;
}

const nlohmann::json* Responses::Failed(const nlohmann::json& event) {
    const auto response = event.find("response");
    if (response == event.end() || !response->is_object()) {
        return nullptr;
    }

    const auto error = response->find("error");
    if (error == response->end() || error->is_null()) {
        return nullptr;
    }
    return &*response;
}

nlohmann::json Responses::Encode(const Config& config, const Request& request, bool reasoning) {
    nlohmann::json body = {
        {"model", request.model},
        {"stream", true},
    };
    body["store"] = false;

    if (Provider::Profile::Find(config.kind).promptCacheKey && !request.sessionId.empty()) {
        body["prompt_cache_key"] = request.sessionId;
    }
    body["include"] = {"reasoning.encrypted_content"};

    if (reasoning && config.requestReasoning) {
        body["reasoning"] = {{"summary", "auto"}};
    }
    if (!request.reasoningEffort.empty()) {
        body["reasoning"]["effort"] = request.reasoningEffort;
    }
    if (!request.reasoningContext.empty()) {
        body["reasoning"]["context"] = request.reasoningContext;
    }

    body["input"] = EncodeInput(config, request);

    if (!request.tools.empty()) {
        body["tools"] = EncodeTools(request.tools);
        if (!request.allowTools) {
            body["tool_choice"] = "none";
        }
    }

    if (request.maxOutputTokens) {
        body["max_output_tokens"] = request.maxOutputTokens;
    }

    return body;
}

std::unique_ptr<EventStream> Responses::Decoder() {
    return std::make_unique<ResponseStream>();
}

}  // namespace Jetstream::Agent::Detail
