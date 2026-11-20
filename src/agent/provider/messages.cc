#include "support/base.hh"

#include "tools/text.hh"

#include <algorithm>
#include <array>
#include <map>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Jetstream::Agent::Detail {

namespace {

using Config = Provider::Config;
using Request = Provider::Request;
using Kind = Provider::Error::Kind;

constexpr std::array<std::string_view, 5> kEfforts{"low", "medium", "high", "xhigh", "max"};

nlohmann::json Ephemeral() {
    return {{"type", "ephemeral"}};
}

bool Thinking(const nlohmann::json& block) {
    const auto type = Wire::Field(block, "type");
    return type == "thinking" || type == "redacted_thinking";
}

void AppendText(nlohmann::json& content, const std::string& text) {
    if (!Text::IsBlank(text)) {
        content.push_back({
            {"type", "text"},
            {"text", text},
        });
    }
}

void AppendImages(nlohmann::json& content, const std::vector<std::string>& images) {
    for (const auto& image : images) {
        content.push_back(Messages::Image(image));
    }
}

nlohmann::json ToolInput(const std::string& arguments) {
    auto input = nlohmann::json::parse(arguments, nullptr, false);
    if (!input.is_object()) {
        return nlohmann::json::object();
    }
    return input;
}

nlohmann::json EncodeToolResult(const Provider::Message& message) {
    nlohmann::json result = {
        {"type", "tool_result"},
        {"tool_use_id", message.toolCallId},
    };

    auto parts = nlohmann::json::array();
    AppendText(parts, message.content);
    AppendImages(parts, message.images);
    if (!parts.empty()) {
        result["content"] = std::move(parts);
    }
    if (message.error) {
        result["is_error"] = true;
    }
    return nlohmann::json::array({std::move(result)});
}

nlohmann::json EncodeContent(const Provider::Message& message) {
    if (message.role == "tool") {
        return EncodeToolResult(message);
    }

    auto content = nlohmann::json::array();
    AppendText(content, message.content);
    AppendImages(content, message.images);
    for (const auto& call : message.toolCalls) {
        content.push_back({
            {"type", "tool_use"},
            {"id", call.id},
            {"name", call.name},
            {"input", ToolInput(call.arguments)},
        });
    }
    return content;
}

std::string ToolName(const nlohmann::json& request, const std::string& id) {
    for (const auto& block : request["content"]) {
        if (Wire::Field(block, "type") == "tool_use" && Wire::Field(block, "id") == id) {
            return Wire::Field(block, "name");
        }
    }
    return {};
}

bool Attach(nlohmann::json& messages, const Provider::Message& message, std::unordered_set<std::string>& attached) {
    if (message.role != "user" || messages.size() < 2 || messages.back()["role"] != "user") {
        return false;
    }

    const auto& request = messages[messages.size() - 2];
    for (auto& block : messages.back()["content"]) {
        const auto id = Wire::Field(block, "tool_use_id");
        if (Wire::Field(block, "type") != "tool_result" || attached.contains(id)) {
            continue;
        }

        const auto name = ToolName(request, id);
        const Provider::FunctionCall call{id, name, {}};
        if (name.empty() || !message.content.starts_with(call.caption())) {
            continue;
        }

        auto& parts = block["content"];
        if (!parts.is_array()) {
            parts = nlohmann::json::array();
        }
        AppendText(parts, message.content);
        AppendImages(parts, message.images);
        attached.insert(id);
        return true;
    }
    return false;
}

void Append(nlohmann::json& messages, const std::string& role, nlohmann::json content) {
    if (content.empty()) {
        return;
    }

    if (!messages.empty() && messages.back()["role"] == role) {
        auto& existing = messages.back()["content"];
        for (auto& block : content) {
            existing.push_back(std::move(block));
        }
        return;
    }

    messages.push_back({
        {"role", role},
        {"content", std::move(content)},
    });
}

void MarkCacheBreakpoint(nlohmann::json& messages) {
    for (auto message = messages.rbegin(); message != messages.rend(); ++message) {
        auto& content = (*message)["content"];
        for (auto block = content.rbegin(); block != content.rend(); ++block) {
            if (block->is_object() && !Thinking(*block)) {
                (*block)["cache_control"] = Ephemeral();
                return;
            }
        }
    }
}

void EncodeConversation(const Config& config, const Request& request, nlohmann::json& body) {
    std::string system;
    auto messages = nlohmann::json::array();
    std::unordered_set<std::string> attached;

    for (const auto& message : request.messages) {
        if (message.role == "system") {
            Text::AppendParagraph(system, message.content);
            continue;
        }

        const auto state = Wire::Replay(message, config.host, config.path, request.model);
        if (state.is_array() && !state.empty()) {
            Append(messages, "assistant", state);
            continue;
        }

        if (Attach(messages, message, attached)) {
            continue;
        }
        Append(messages, message.role == "assistant" ? "assistant" : "user", EncodeContent(message));
    }

    MarkCacheBreakpoint(messages);

    if (!system.empty()) {
        body["system"] = nlohmann::json::array({{
            {"type", "text"},
            {"text", std::move(system)},
            {"cache_control", Ephemeral()},
        }});
    }
    body["messages"] = std::move(messages);
}

nlohmann::json EncodeTools(const std::span<const Provider::Function> tools) {
    auto encoded = nlohmann::json::array();
    for (const auto& tool : tools) {
        auto function = Wire::Function(tool);
        encoded.push_back({
            {"name", std::move(function["name"])},
            {"description", std::move(function["description"])},
            {"input_schema", std::move(function["parameters"])},
        });
    }
    return encoded;
}

void EncodeThinking(const Config& config, const Request& request, bool reasoning, nlohmann::json& body) {
    if (request.reasoningEffort.empty()) {
        return;
    }
    if (request.reasoningEffort == "none") {
        body["thinking"] = {{"type", "disabled"}};
        return;
    }

    body["thinking"] = {{"type", "adaptive"}};
    if (reasoning && config.requestReasoning) {
        body["thinking"]["display"] = "summarized";
    }
    if (const auto effort = Messages::Effort(request.reasoningEffort); !effort.empty()) {
        body["output_config"] = {{"effort", effort}};
    }
}

I64 Index(const nlohmann::json& chunk) {
    const auto index = chunk.find("index");
    if (index == chunk.end() || !index->is_number_integer()) {
        return 0;
    }
    return index->get<I64>();
}

void Extend(nlohmann::json& block, const char* key, const std::string& value) {
    block[key] = Wire::Field(block, key) + value;
}

class MessageStream final : public EventStream {
 public:
    std::string payload() const override {
        auto content = nlohmann::json::array();
        for (const auto& [index, block] : blocks) {
            const bool emptyText = Wire::Field(block, "type") == "text" && Text::IsBlank(Wire::Field(block, "text"));
            if (!emptyText) {
                content.push_back(block);
            }
        }

        if (content.empty()) {
            return "";
        }
        return content.dump();
    }

 protected:
    bool decode(const nlohmann::json& chunk) override {
        using Handler = bool (MessageStream::*)(const nlohmann::json&);
        static const std::unordered_map<std::string, Handler> handlers{
            {"message_start", &MessageStream::messageStart},
            {"content_block_start", &MessageStream::blockStart},
            {"content_block_delta", &MessageStream::blockDelta},
            {"content_block_stop", &MessageStream::blockStop},
            {"message_delta", &MessageStream::messageDelta},
            {"message_stop", &MessageStream::messageStop},
            {"error", &MessageStream::failed},
        };

        const auto handler = handlers.find(Wire::Field(chunk, "type"));
        if (handler == handlers.end()) {
            return true;
        }
        return (this->*handler->second)(chunk);
    }

    bool complete() override {
        if (malformed) {
            return fail(Kind::Protocol, "The provider returned malformed tool call arguments.");
        }

        for (const auto& [index, block] : blocks) {
            if (Wire::Field(block, "type") != "tool_use") {
                continue;
            }

            const auto input = block.find("input");
            Provider::FunctionCall call{
                Wire::Field(block, "id"),
                Wire::Field(block, "name"),
                input != block.end() && input->is_object() ? input->dump() : "{}",
            };
            if (call.id.empty() || call.name.empty()) {
                return fail(Kind::Protocol, "The provider returned an incomplete tool call.");
            }
            toolCalls.push_back(std::move(call));
        }
        return true;
    }

 private:
    bool messageStart(const nlohmann::json& chunk) {
        sawChoice = true;

        const auto message = chunk.find("message");
        if (message != chunk.end() && message->is_object()) {
            countUsage(*message);
        }
        return true;
    }

    bool blockStart(const nlohmann::json& chunk) {
        const auto block = chunk.find("content_block");
        if (block == chunk.end() || !block->is_object()) {
            return fail(Kind::Protocol, "The provider returned an invalid content block.");
        }

        const auto position = Index(chunk);
        auto& stored = blocks[position] = *block;
        const auto type = Wire::Field(stored, "type");

        if (type == "text") {
            token(Wire::Field(stored, "text"));
        } else if (type == "thinking") {
            const auto value = Wire::Field(stored, "thinking");
            outputStarted |= !value.empty();
            reasoning(value, position, true);
        } else if (type == "tool_use") {
            stored["input"] = nlohmann::json::object();
            arguments[position].clear();
            announce(Wire::Field(stored, "name"));
        } else {
            outputStarted = true;
        }
        return true;
    }

    bool blockDelta(const nlohmann::json& chunk) {
        const auto position = Index(chunk);
        const auto block = blocks.find(position);
        const auto delta = chunk.find("delta");
        if (block == blocks.end() || delta == chunk.end() || !delta->is_object()) {
            return fail(Kind::Protocol, "The provider streamed an invalid content block delta.");
        }

        const auto type = Wire::Field(*delta, "type");
        if (type == "text_delta") {
            const auto value = Wire::Field(*delta, "text");
            Extend(block->second, "text", value);
            token(value);
        } else if (type == "thinking_delta") {
            const auto value = Wire::Field(*delta, "thinking");
            Extend(block->second, "thinking", value);
            outputStarted |= !value.empty();
            reasoning(value, position, true);
        } else if (type == "signature_delta") {
            Extend(block->second, "signature", Wire::Field(*delta, "signature"));
            outputStarted = true;
        } else if (type == "input_json_delta") {
            arguments[position] += Wire::Field(*delta, "partial_json");
            outputStarted = true;
        }
        return true;
    }

    bool blockStop(const nlohmann::json& chunk) {
        const auto position = Index(chunk);
        const auto block = blocks.find(position);
        if (block == blocks.end() || Wire::Field(block->second, "type") != "tool_use") {
            return true;
        }

        const auto& json = arguments[position];
        auto input = json.empty() ? nlohmann::json::object() : nlohmann::json::parse(json, nullptr, false);
        if (!input.is_object()) {
            malformed = true;
            return true;
        }
        block->second["input"] = std::move(input);
        return true;
    }

    bool messageDelta(const nlohmann::json& chunk) {
        countUsage(chunk);

        const auto delta = chunk.find("delta");
        if (delta == chunk.end() || !delta->is_object()) {
            return true;
        }

        const auto reason = Wire::Field(*delta, "stop_reason");
        if (reason == "max_tokens" || reason == "model_context_window_exceeded") {
            truncated = true;
        } else if (reason == "refusal") {
            return fail(Kind::InvalidRequest, "The provider stopped the response: refusal");
        }
        return true;
    }

    bool messageStop(const nlohmann::json&) {
        finishedChoice = true;
        return true;
    }

    bool failed(const nlohmann::json& chunk) {
        error = Diagnostics::Payload(chunk);
        return false;
    }

    void countUsage(const nlohmann::json& object) {
        const auto found = object.find("usage");
        if (found == object.end() || !found->is_object()) {
            return;
        }

        const auto read = [&](const char* key, U64& target) {
            const auto value = found->find(key);
            if (value != found->end() && value->is_number_unsigned()) {
                target = value->get<U64>();
            }
        };
        read("input_tokens", input);
        read("output_tokens", output);
        read("cache_read_input_tokens", cacheRead);
        read("cache_creation_input_tokens", cacheWrite);

        usage.inputTokens = input + cacheRead + cacheWrite;
        usage.outputTokens = output;
        usage.cachedTokens = cacheRead;
        usage.cacheWriteTokens = cacheWrite;

        const auto details = found->find("output_tokens_details");
        if (details != found->end() && details->is_object()) {
            const auto thinking = details->find("thinking_tokens");
            if (thinking != details->end() && thinking->is_number_unsigned()) {
                usage.reasoningTokens = thinking->get<U64>();
            }
        }
    }

    std::map<I64, nlohmann::json> blocks;
    std::map<I64, std::string> arguments;
    U64 input = 0;
    U64 output = 0;
    U64 cacheRead = 0;
    U64 cacheWrite = 0;
    bool malformed = false;
};

}  // namespace

std::string Messages::Effort(const std::string& effort) {
    if (effort == "minimal") {
        return "low";
    }
    if (std::find(kEfforts.begin(), kEfforts.end(), effort) != kEfforts.end()) {
        return effort;
    }
    return {};
}

nlohmann::json Messages::Image(const std::string& url) {
    constexpr std::string_view prefix = "data:";
    constexpr std::string_view marker = ";base64,";

    const auto separator = url.find(marker);
    if (!url.starts_with(prefix) || separator == std::string::npos) {
        return {
            {"type", "image"},
            {"source", {{"type", "url"}, {"url", url}}},
        };
    }

    return {
        {"type", "image"},
        {"source", {
            {"type", "base64"},
            {"media_type", url.substr(prefix.size(), separator - prefix.size())},
            {"data", url.substr(separator + marker.size())},
        }},
    };
}

bool Messages::DropThinking(nlohmann::json& body) {
    const auto messages = body.find("messages");
    if (messages == body.end() || !messages->is_array()) {
        return false;
    }

    bool dropped = false;
    for (auto message = messages->begin(); message != messages->end();) {
        auto& content = (*message)["content"];
        if (content.is_array()) {
            const auto size = content.size();
            content.erase(std::remove_if(content.begin(), content.end(), Thinking), content.end());
            dropped |= content.size() != size;
        }

        if (content.is_array() && content.empty()) {
            message = messages->erase(message);
        } else {
            ++message;
        }
    }
    return dropped;
}

nlohmann::json Messages::Encode(const Config& config, const Request& request, bool reasoning) {
    nlohmann::json body = {
        {"model", request.model},
        {"stream", true},
    };
    body["max_tokens"] = request.maxOutputTokens ? request.maxOutputTokens : DefaultMaxTokens;

    EncodeConversation(config, request, body);
    EncodeThinking(config, request, reasoning, body);

    if (!request.tools.empty()) {
        body["tools"] = EncodeTools(request.tools);
        if (!request.allowTools) {
            body["tool_choice"] = {{"type", "none"}};
        }
    }

    return body;
}

std::unique_ptr<EventStream> Messages::Decoder() {
    return std::make_unique<MessageStream>();
}

}  // namespace Jetstream::Agent::Detail
