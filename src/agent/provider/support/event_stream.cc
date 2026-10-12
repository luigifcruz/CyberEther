#include "base.hh"

#include <utility>

namespace Jetstream::Agent::Detail {

bool EventStream::append(std::string_view bytes) {
    pending.append(bytes);

    size_t newline;
    while ((newline = pending.find('\n')) != std::string::npos) {
        auto value = pending.substr(0, newline);
        pending.erase(0, newline + 1);
        if (!line(std::move(value))) {
            return false;
        }
    }
    return true;
}

bool EventStream::finish() {
    if (error) {
        return false;
    }
    if (!pending.empty() && !line(std::exchange(pending, {}))) {
        return false;
    }
    if (!dispatch()) {
        return false;
    }

    if (!sawChoice) {
        return fail(Provider::Error::Kind::Protocol, "The provider returned no completion data.");
    }
    if (!done && !finishedChoice) {
        return fail(Provider::Error::Kind::Protocol, "The response stream ended before completion.");
    }

    if (truncated) {
        return fail(Provider::Error::Kind::OutputLimit, Truncated);
    }
    return complete();
}

bool EventStream::feed(const nlohmann::json& event) {
    return decode(event);
}

bool EventStream::settle() {
    return !error && complete();
}

bool EventStream::fail(Provider::Error::Kind kind, std::string message) {
    error = {kind, 0, std::move(message)};
    return false;
}

void EventStream::token(const std::string& value) {
    outputStarted |= !value.empty();
    text += value;
    if (!value.empty() && onToken) {
        onToken(value);
    }
}

void EventStream::announce(const std::string& name) {
    outputStarted = true;
    if (!name.empty() && onToolCall) {
        onToolCall(name);
    }
}

void EventStream::reasoning(const std::string& value, I64 index, bool summary) {
    if (value.empty() || !onReasoning) {
        return;
    }

    const bool newBlock = index != lastReasoningIndex;
    const bool needsBreak = lastReasoningChar != '\0' && lastReasoningChar != '\n' && value.front() != '\n';
    if (newBlock && needsBreak) {
        onReasoning("\n\n", summary);
    }

    lastReasoningIndex = index;
    lastReasoningChar = value.back();
    onReasoning(value, summary);
}

void EventStream::readUsage(const nlohmann::json& object, const UsageKeys& keys) {
    const auto found = object.find("usage");
    if (found == object.end() || !found->is_object()) {
        return;
    }

    const auto count = [&](const char* key) -> U64 {
        const auto value = found->find(key);
        if (value == found->end() || !value->is_number_unsigned()) {
            return 0;
        }
        return value->get<U64>();
    };
    const auto detail = [&](const char* object, const char* key) -> U64 {
        const auto details = found->find(object);
        if (details == found->end() || !details->is_object()) {
            return 0;
        }

        const auto value = details->find(key);
        if (value == details->end() || !value->is_number_unsigned()) {
            return 0;
        }
        return value->get<U64>();
    };

    usage = {count(keys.input), count(keys.output)};
    usage.cachedTokens = detail(keys.inputDetails, "cached_tokens");
    usage.cacheWriteTokens = detail(keys.inputDetails, "cache_write_tokens");
    usage.reasoningTokens = detail(keys.outputDetails, "reasoning_tokens");

    if (const auto cost = found->find("cost"); cost != found->end() && cost->is_number()) {
        usage.cost = cost->get<F64>();
    }
}

bool EventStream::line(std::string value) {
    if (!value.empty() && value.back() == '\r') {
        value.pop_back();
    }

    if (value.empty()) {
        return dispatch();
    }
    if (value.front() == ':') {
        return true;
    }

    const auto colon = value.find(':');
    const auto field = value.substr(0, colon);

    std::string content;
    if (colon != std::string::npos) {
        content = value.substr(colon + 1);
    }
    if (!content.empty() && content.front() == ' ') {
        content.erase(0, 1);
    }

    if (field == "event") {
        event = std::move(content);
    } else if (field == "data") {
        if (!data.empty()) {
            data += '\n';
        }
        data += content;
    }
    return true;
}

bool EventStream::dispatch() {
    if (data.empty() || done) {
        data.clear();
        event.clear();
        return true;
    }

    auto payload = std::exchange(data, {});
    const auto eventName = std::exchange(event, {});
    if (payload == "[DONE]") {
        done = true;
        return true;
    }

    const auto chunk = nlohmann::json::parse(payload, nullptr, false);
    if (!chunk.is_object()) {
        return fail(Provider::Error::Kind::Protocol,
                    "The provider returned malformed JSON in the response stream.");
    }

    const bool errorType = chunk.contains("type") && chunk["type"] == "error";
    if (eventName == "error" || chunk.contains("error") || errorType) {
        error = Diagnostics::Payload(chunk);
        return false;
    }

    return decode(chunk);
}

}  // namespace Jetstream::Agent::Detail
