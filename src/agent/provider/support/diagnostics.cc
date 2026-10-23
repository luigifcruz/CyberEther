#include "base.hh"

#include "jetstream/logger.hh"

#include <algorithm>
#include <array>
#include <cctype>

namespace Jetstream::Agent::Detail {

namespace {

using Kind = Provider::Error::Kind;

struct Classification {
    Kind kind;
    std::string_view summary;
    std::string_view advice;
};

struct Rule {
    I32 status;
    I32 alternateStatus;
    std::string_view code;
    Classification classification;
};

constexpr std::array kRules{
    Rule{402, 0, "insufficient_quota",
         {Kind::Quota, "Quota or credits exhausted", "Check billing and usage limits."}},
    Rule{401, 0, "invalid_api_key",
         {Kind::Authentication, "Authentication failed", "Check the configured API key."}},
    Rule{403, 0, "",
         {Kind::Permission, "Access denied", "Check account permissions and model access."}},
    Rule{404, 0, "model_not_found",
         {Kind::NotFound, "Endpoint or model not found", "Check the API path and model id."}},
    Rule{429, 0, "rate_limit_exceeded",
         {Kind::RateLimit, "Rate limit exceeded", "Try again later."}},
    Rule{408, 504, "",
         {Kind::Timeout, "The provider timed out", "Try again later."}},
};

constexpr std::array<std::string_view, 4> kServerCodes{
    "server_error", "internal_server_error", "overloaded_error", "service_unavailable",
};

void ReadErrorCode(const nlohmann::json& payload, Provider::Error& error) {
    if (!payload.is_object()) {
        return;
    }
    const auto found = payload.find("error");
    const auto& value = found != payload.end() ? *found : payload;
    if (!value.is_object()) {
        return;
    }
    error.code = Wire::Field(value, "code");
    if (error.code.empty()) {
        error.code = Wire::Field(value, "type");
    }
    const auto code = value.find("code");
    if (error.status == 0 && code != value.end() && code->is_number_integer()) {
        const auto numeric = code->get<I64>();
        if (numeric >= 400 && numeric <= 599) {
            error.status = static_cast<I32>(numeric);
        }
    }
}

bool Matches(const Rule& rule, const Provider::Error& error) {
    return error.status == rule.status ||
           (rule.alternateStatus != 0 && error.status == rule.alternateStatus) ||
           (!rule.code.empty() && error.code == rule.code);
}

Classification Classify(const Provider::Error& error) {
    for (const auto& rule : kRules) {
        if (Matches(rule, error)) {
            return rule.classification;
        }
    }
    const bool serverCode = std::find(kServerCodes.begin(), kServerCodes.end(), error.code) !=
                            kServerCodes.end();
    if (error.status >= 500 || (error.status == 0 && serverCode)) {
        return {Kind::Server, error.status == 503 ? "Service unavailable" : "Provider request failed",
                "Try again later."};
    }
    if (error.status == 0) {
        return {Kind::InvalidRequest, "Provider request failed", ""};
    }
    return {Kind::InvalidRequest, "Request rejected", ""};
}

std::string CleanDetail(const std::string& detail) {
    auto cleaned = Diagnostics::Flatten(detail);
    constexpr std::string_view upstreamPrefix = "Upstream request failed:";
    if (cleaned.starts_with(upstreamPrefix)) {
        cleaned = Diagnostics::Flatten(std::string_view(cleaned).substr(upstreamPrefix.size()));
    }
    return cleaned;
}

std::string RetryHint(const Provider::Error& error) {
    const auto retry = Diagnostics::Flatten(error.retryAfter);
    if (retry.empty() || (error.kind != Kind::RateLimit && error.kind != Kind::Server &&
                          error.kind != Kind::Timeout)) {
        return {};
    }
    const bool seconds = std::all_of(retry.begin(), retry.end(),
        [](unsigned char value) { return std::isdigit(value); });
    return seconds ? jst::fmt::format(" Retry in {}s.", retry)
                   : jst::fmt::format(" Retry after {}.", retry);
}

std::string Describe(const Provider::Error& error, const std::string& detail,
                     const Classification& classification) {
    std::string message = detail.empty() ? std::string(classification.summary) : detail;
    if (message.ends_with('.')) {
        message.pop_back();
    }
    if (error.status > 0) {
        message += jst::fmt::format(" (HTTP {})", error.status);
    }
    message += '.';
    if (const auto hint = RetryHint(error); !hint.empty()) {
        message += hint;
    } else if (detail.empty() && !classification.advice.empty()) {
        message += " " + std::string(classification.advice);
    }
    return message;
}

bool ContextOverflow(const Provider::Error& error, std::string detail) {
    if (error.code == "context_length_exceeded") {
        return true;
    }
    std::transform(detail.begin(), detail.end(), detail.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return detail.find("context length") != std::string::npos ||
           detail.find("context window") != std::string::npos ||
           detail.find("maximum context") != std::string::npos ||
           detail.find("prompt is too long") != std::string::npos;
}

}  // namespace

std::string Diagnostics::Flatten(std::string_view value) {
    std::string result;
    bool space = false;
    for (const unsigned char character : value) {
        if (std::isspace(character)) {
            space = !result.empty();
            continue;
        }
        if (space) {
            result += ' ';
        }
        result += static_cast<char>(character);
        space = false;
    }
    return result;
}

std::string Diagnostics::Redact(std::string text, std::string_view secret) {
    if (secret.empty()) {
        return text;
    }
    size_t position = 0;
    while ((position = text.find(secret, position)) != std::string::npos) {
        text.replace(position, secret.size(), "[redacted]");
        position += 10;
    }
    return text;
}

void Diagnostics::Sanitize(Provider::Error& failure, std::string_view secret) {
    failure.message = Redact(Flatten(failure.message), secret);
    if (failure.message.size() > MaxLength) {
        failure.message = failure.message.substr(0, MaxLength) + "...";
    }
    failure.code = Redact(std::move(failure.code), secret);
    failure.retryAfter = Redact(std::move(failure.retryAfter), secret);
}

Result Diagnostics::Report(Provider::Error failure, Provider::Error* output, const std::string& context,
                           std::string_view secret) {
    failure.message = context + ": " + failure.message;
    Sanitize(failure, secret);
    if (failure.kind != Provider::Error::Kind::Cancelled) {
        JST_ERROR("[AGENT_CLIENT] {}", failure.message);
    }
    if (output) {
        *output = std::move(failure);
    }
    return Result::ERROR;
}

std::string Diagnostics::MissingKey() {
    return "No API key is stored for this provider.";
}

Provider::Error Diagnostics::Payload(const nlohmann::json& payload, I32 status, const std::string& retryAfter) {
    Provider::Error error;
    error.status = status;
    error.retryAfter = retryAfter;
    ReadErrorCode(payload, error);
    const auto classification = Classify(error);
    error.kind = classification.kind;
    const auto detail = CleanDetail(Wire::Error(payload));
    error.message = Describe(error, detail, classification);
    if (ContextOverflow(error, detail)) {
        error.kind = Provider::Error::Kind::ContextLength;
        error.message += " The request exceeds the model's context window.";
    }
    return error;
}

Provider::Error Diagnostics::Http(I32 status, const std::string& body, const std::string& retryAfter) {
    return Payload(nlohmann::json::parse(body, nullptr, false), status, retryAfter);
}

Provider::Error Diagnostics::Transport(const Platform::Http::Error& failure) {
    using Failure = Platform::Http::Error::Kind;

    Provider::Error error;
    switch (failure.kind) {
        case Failure::Timeout:
            error.kind = Kind::Timeout;
            error.message = "Request timed out. Try again later.";
            break;
        case Failure::Tls:
            error.kind = Kind::Tls;
            error.message = failure.message + ". Check the endpoint certificate.";
            break;
        case Failure::Unsupported:
            error.kind = Kind::Unsupported;
            error.message = failure.message + ".";
            break;
        case Failure::Cancelled:
            error.kind = Kind::Cancelled;
            error.message = "Request cancelled.";
            break;
        default:
            error.kind = Kind::Connection;
            error.message = failure.message + ". Check the endpoint and network connection.";
            break;
    }
    return error;
}

Provider::Error Diagnostics::Response(const Platform::Http::Client::Response& response, nlohmann::json& body) {
    if (response.error) {
        return Transport(response.error);
    }
    if (response.status < 200 || response.status >= 300) {
        return Http(response.status, response.body, response.header("Retry-After"));
    }
    body = nlohmann::json::parse(response.body, nullptr, false);
    if (body.is_object() && body.contains("error")) {
        return Payload(body);
    }
    return {};
}

}  // namespace Jetstream::Agent::Detail
