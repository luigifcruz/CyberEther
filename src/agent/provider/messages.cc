#include "support/base.hh"

#include <map>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace Jetstream::Agent::Detail {

namespace {

using Config = Provider::Config;
using Request = Provider::Request;
using Kind = Provider::Error::Kind;

const nlohmann::json kCacheBreakpoint = {{"type", "ephemeral"}};

nlohmann::json TextBlock(const std::string& text) {
    return {{"type", "text"}, {"text", text}};
}

bool Replayable(const nlohmann::json& block) {
    if (!block.is_object()) {
        return false;
    }
    const auto type = Wire::Field(block, "type");
    if (type == "text") {
        return !Wire::Field(block, "text").empty();
    }
    return type == "thinking" || type == "redacted_thinking" || type == "tool_use";
}

bool Cacheable(const nlohmann::json& block) {
    const auto type = Wire::Field(block, "type");
    return block.is_object() && type != "thinking" && type != "redacted_thinking";
}

nlohmann::json ToolResult(const Provider::Message& message) {
    nlohmann::json block = {
        {"type", "tool_result"},
        {"tool_use_id", message.toolCallId},
    };
    if (message.images.empty()) {
        block["content"] = message.content;
    } else {
        auto parts = nlohmann::json::array();
        if (!message.content.empty()) {
            parts.push_back(TextBlock(message.content));
        }
        for (const auto& image : message.images) {
            parts.push_back(Messages::Image(image));
        }
        block["content"] = std::move(parts);
    }
    if (message.error) {
        block["is_error"] = true;
    }
    return block;
}

nlohmann::json UserContent(const Provider::Message& message) {
    auto parts = nlohmann::json::array();
    if (!message.content.empty()) {
        parts.push_back(TextBlock(message.content));
    }
    for (const auto& image : message.images) {
        parts.push_back(Messages::Image(image));
    }
    return parts;
}

nlohmann::json AssistantContent(const Provider::Message& message, const nlohmann::json& state) {
    auto parts = nlohmann::json::array();
    if (state.is_array()) {
        for (const auto& block : state) {
            if (Replayable(block)) {
                parts.push_back(block);
            }
        }
        if (!parts.empty()) {
            return parts;
        }
    }

    if (!message.content.empty()) {
        parts.push_back(TextBlock(message.content));
    }
    for (const auto& call : message.toolCalls) {
        parts.push_back(Messages::ToolUse(call));
    }
    return parts;
}

void Append(nlohmann::json& messages, const Provider::Message& message, const nlohmann::json& state) {
    std::string role = "user";
    nlohmann::json content;
    if (message.role == "tool") {
        content = nlohmann::json::array({ToolResult(message)});
    } else if (message.role == "assistant") {
        role = "assistant";
        content = AssistantContent(message, state);
    } else {
        content = UserContent(message);
    }

    if (content.empty() || (messages.empty() && role != "user")) {
        return;
    }

    if (!messages.empty() && messages.back()["role"] == role) {
        auto& existing = messages.back()["content"];
        for (auto& block : content) {
            existing.push_back(std::move(block));
        }
        return;
    }
    messages.push_back({{"role", role}, {"content", std::move(content)}});
}

void MarkCacheBreakpoint(nlohmann::json& messages) {
    if (messages.empty()) {
        return;
    }
    auto& content = messages.back()["content"];
    for (auto it = content.rbegin(); it != content.rend(); ++it) {
        if (Cacheable(*it)) {
            (*it)["cache_control"] = kCacheBreakpoint;
            return;
        }
    }
}

I64 BlockIndex(const nlohmann::json& chunk) {
    const auto index = chunk.find("index");
    if (index != chunk.end() && index->is_number_integer()) {
        return index->get<I64>();
    }
    return 0;
}

void AppendField(nlohmann::json& block, const char* name, const std::string& value) {
    auto field = block.find(name);
    if (field == block.end() || !field->is_string()) {
        block[name] = value;
        return;
    }
    field->get_ref<std::string&>() += value;
}

class MessageStream final : public EventStream {
 public:
    std::string payload() const override {
        auto output = nlohmann::json::array();
        for (const auto& [index, block] : blocks) {
            if (Replayable(block)) {
                output.push_back(block);
            }
        }

        if (output.empty()) {
            return "";
        }
        return output.dump();
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
        };

        const auto handler = handlers.find(Wire::Field(chunk, "type"));
        if (handler == handlers.end()) {
            return true;
        }
        return (this->*handler->second)(chunk);
    }

    bool complete() override {
        for (const auto& [index, block] : blocks) {
            if (Wire::Field(block, "type") != "tool_use") {
                continue;
            }

            Provider::FunctionCall call{Wire::Field(block, "id"), Wire::Field(block, "name"), {}};
            if (call.id.empty() || call.name.empty()) {
                return fail(Kind::Protocol, "The provider returned an incomplete tool call.");
            }

            const auto raw = partials.find(index);
            call.arguments = raw != partials.end() && !raw->second.empty() ? raw->second : "{}";
            toolCalls.push_back(std::move(call));
        }
        return true;
    }

 private:
    bool messageStart(const nlohmann::json& chunk) {
        sawChoice = true;
        const auto message = chunk.find("message");
        if (message != chunk.end() && message->is_object()) {
            usageFrom(*message);
        }
        return true;
    }

    bool blockStart(const nlohmann::json& chunk) {
        const auto block = chunk.find("content_block");
        if (block == chunk.end() || !block->is_object()) {
            return true;
        }

        const auto index = BlockIndex(chunk);
        blocks[index] = *block;

        const auto type = Wire::Field(*block, "type");
        if (type == "tool_use") {
            partials[index].clear();
            announce(Wire::Field(*block, "name"));
        } else if (type == "text") {
            token(Wire::Field(*block, "text"));
        } else if (type == "thinking") {
            thought(Wire::Field(*block, "thinking"), index);
        }
        return true;
    }

    bool blockDelta(const nlohmann::json& chunk) {
        const auto delta = chunk.find("delta");
        if (delta == chunk.end() || !delta->is_object()) {
            return true;
        }

        const auto index = BlockIndex(chunk);
        auto& block = blocks[index];
        const auto type = Wire::Field(*delta, "type");
        if (type == "text_delta") {
            const auto value = Wire::Field(*delta, "text");
            AppendField(block, "text", value);
            token(value);
        } else if (type == "thinking_delta") {
            const auto value = Wire::Field(*delta, "thinking");
            AppendField(block, "thinking", value);
            thought(value, index);
        } else if (type == "signature_delta") {
            AppendField(block, "signature", Wire::Field(*delta, "signature"));
        } else if (type == "input_json_delta") {
            partials[index] += Wire::Field(*delta, "partial_json");
            outputStarted = true;
        }
        return true;
    }

    bool blockStop(const nlohmann::json& chunk) {
        const auto found = blocks.find(BlockIndex(chunk));
        if (found == blocks.end() || Wire::Field(found->second, "type") != "tool_use") {
            return true;
        }

        const auto& raw = partials[found->first];
        auto input = raw.empty() ? nlohmann::json::object() : nlohmann::json::parse(raw, nullptr, false);
        if (!input.is_object()) {
            return fail(Kind::Protocol, "The provider returned malformed tool call arguments.");
        }
        found->second["input"] = std::move(input);
        return true;
    }

    bool messageDelta(const nlohmann::json& chunk) {
        usageFrom(chunk);
        finishedChoice = true;

        std::string reason;
        const auto delta = chunk.find("delta");
        if (delta != chunk.end() && delta->is_object()) {
            reason = Wire::Field(*delta, "stop_reason");
        }

        if (reason == "max_tokens") {
            truncated = true;
        } else if (reason == "refusal") {
            return fail(Kind::InvalidRequest, "The model declined to answer this request.");
        }
        return true;
    }

    void thought(const std::string& value, I64 index) {
        outputStarted |= !value.empty();
        reasoning(value, index, true);
    }

    void usageFrom(const nlohmann::json& object) {
        const auto found = object.find("usage");
        if (found == object.end() || !found->is_object()) {
            return;
        }

        const auto count = [&](const char* key, U64& target) {
            const auto value = found->find(key);
            if (value != found->end() && value->is_number_unsigned()) {
                target = value->get<U64>();
            }
        };

        count("output_tokens", usage.outputTokens);
        count("cache_read_input_tokens", usage.cachedTokens);
        count("cache_creation_input_tokens", usage.cacheWriteTokens);

        U64 uncached = 0;
        count("input_tokens", uncached);
        if (found->contains("input_tokens")) {
            usage.inputTokens = uncached + usage.cachedTokens + usage.cacheWriteTokens;
        }
    }

    std::map<I64, nlohmann::json> blocks;
    std::map<I64, std::string> partials;
};

}  // namespace

nlohmann::json Messages::Tool(const Provider::Function& tool) {
    auto schema = nlohmann::json::parse(tool.parametersJson, nullptr, false);
    if (!schema.is_object()) {
        schema = {{"type", "object"}, {"properties", nlohmann::json::object()}};
    }
    if (!schema.contains("type")) {
        schema["type"] = "object";
    }
    return {
        {"name", tool.name},
        {"description", tool.description},
        {"input_schema", std::move(schema)},
    };
}

nlohmann::json Messages::Image(const std::string& url) {
    constexpr std::string_view prefix = "data:";
    const auto comma = url.find(',');
    if (!url.starts_with(prefix) || comma == std::string::npos) {
        return {{"type", "image"}, {"source", {{"type", "url"}, {"url", url}}}};
    }

    const auto header = url.substr(prefix.size(), comma - prefix.size());
    const auto mediaType = header.substr(0, header.find(';'));
    return {
        {"type", "image"},
        {"source", {
            {"type", "base64"},
            {"media_type", mediaType},
            {"data", url.substr(comma + 1)},
        }},
    };
}

nlohmann::json Messages::ToolUse(const Provider::FunctionCall& call) {
    auto input = nlohmann::json::parse(call.arguments, nullptr, false);
    if (!input.is_object()) {
        input = nlohmann::json::object();
    }
    return {
        {"type", "tool_use"},
        {"id", call.id},
        {"name", call.name},
        {"input", std::move(input)},
    };
}

nlohmann::json Messages::Encode(const Config& config, const Request& request, bool reasoning) {
    nlohmann::json body = {
        {"model", request.model},
        {"stream", true},
        {"max_tokens", request.maxOutputTokens ? request.maxOutputTokens : DefaultMaxTokens},
    };

    std::string system;
    auto messages = nlohmann::json::array();
    for (const auto& message : request.messages) {
        if (message.role == "system") {
            if (!system.empty() && !message.content.empty()) {
                system += "\n\n";
            }
            system += message.content;
            continue;
        }

        const auto state = Wire::Replay(message, config.host, config.path, request.model);
        if (Wire::Empty(message, state)) {
            continue;
        }
        Append(messages, message, state);
    }

    if (!system.empty()) {
        auto block = TextBlock(system);
        block["cache_control"] = kCacheBreakpoint;
        body["system"] = nlohmann::json::array({std::move(block)});
    }
    MarkCacheBreakpoint(messages);
    body["messages"] = std::move(messages);

    body["thinking"] = {{"type", "adaptive"}};
    if (reasoning && config.requestReasoning) {
        body["thinking"]["display"] = "summarized";
    }
    if (!request.reasoningEffort.empty()) {
        body["output_config"] = {{"effort", request.reasoningEffort}};
    }

    if (!request.tools.empty()) {
        auto tools = nlohmann::json::array();
        for (const auto& tool : request.tools) {
            tools.push_back(Tool(tool));
        }
        body["tools"] = std::move(tools);
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
