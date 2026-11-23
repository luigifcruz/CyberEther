#include "support/base.hh"

#include "jetstream/logger.hh"
#include "tools/random.hh"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace Jetstream::Agent {

namespace {

using Config = Provider::Config;
using Detail::Wire;

std::string ProviderContext(const Config& config) {
    if (config.kind != Provider::Kind::Custom) {
        return Provider::Profile::Find(config.kind).name;
    }
    if (config.host.empty()) {
        return "Provider";
    }
    return config.host;
}

Result Fail(Provider::Error failure, Provider::Error* output, const Config& config, std::string_view key,
            const std::string& suffix = {}) {
    return Detail::Diagnostics::Report(std::move(failure), output, ProviderContext(config) + suffix, key);
}

std::string ReadKey(const Config& config) {
    return config.apiKey ? config.apiKey() : std::string{};
}

constexpr U32 kReasoningSetting = 1;
constexpr U32 kEncryptedContent = 2;
constexpr U32 kThinking = 4;
constexpr U32 kReasoningEffort = 8;

void Remove(nlohmann::json& request, U32 rejected) {
    if (rejected & kReasoningSetting) {
        request.erase("reasoning");
    }
    if (rejected & kEncryptedContent) {
        request.erase("include");
    }
    if (rejected & kThinking) {
        request.erase("thinking");
        request.erase("output_config");
    }
    if (rejected & kReasoningEffort) {
        request.erase("reasoning_effort");
    }
}

std::string RejectedParam(const std::string& response) {
    const auto payload = nlohmann::json::parse(response, nullptr, false);
    const bool hasError = payload.is_object() && payload.contains("error") && payload["error"].is_object();
    if (hasError) {
        return Wire::Field(payload["error"], "param");
    }
    return {};
}

U32 ParamOptions(std::string_view param, U32 requested) {
    constexpr std::string_view bodyPrefix = "body.";
    if (param.starts_with(bodyPrefix)) {
        param.remove_prefix(bodyPrefix.size());
    }

    if (param == "reasoning_effort") {
        return requested & kReasoningEffort;
    }
    if (param == "reasoning" || param.starts_with("reasoning.")) {
        return requested & kReasoningSetting;
    }
    if (param == "include" || param.starts_with("include[")) {
        return requested & kEncryptedContent;
    }
    if (param.starts_with("thinking") || param.starts_with("output_config")) {
        return requested & kThinking;
    }
    return 0;
}

std::string Lowercase(std::string message) {
    std::transform(message.begin(), message.end(), message.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return message;
}

U32 ThinkingOptions(const std::string& message) {
    const auto mentions = [&](const char* phrase) {
        return message.find(phrase) != std::string::npos;
    };

    const bool named = mentions("thinking") || mentions("output_config") || mentions("effort");
    const bool rejected = mentions("not supported") ||
                          mentions("unsupported") ||
                          mentions("does not support") ||
                          mentions("extra inputs") ||
                          mentions("input should be") ||
                          mentions("between_tools");
    return named && rejected ? kThinking : 0;
}

U32 MessageOptions(std::string message, U32 requested) {
    const auto mentions = [&](const char* phrase) {
        return message.find(phrase) != std::string::npos;
    };

    const bool rejected = mentions("unsupported") ||
                          mentions("not supported") ||
                          mentions("unknown parameter") ||
                          mentions("unrecognized");
    if (!rejected) {
        return 0;
    }

    U32 mentioned = 0;
    if (requested & kReasoningEffort) {
        constexpr std::string_view effortField = "reasoning_effort";
        for (auto at = message.find(effortField); at != std::string::npos; at = message.find(effortField)) {
            message.erase(at, effortField.size());
            mentioned |= kReasoningEffort;
        }
    }

    const bool encrypted = mentions("reasoning.encrypted_content");

    if (encrypted || mentions("include")) {
        mentioned |= kEncryptedContent;
    }
    if (mentions("reasoning") && (!encrypted || mentions("reasoning.summary"))) {
        mentioned |= kReasoningSetting;
    }
    return mentioned;
}

U32 Rejected(const Provider::Error& failure, const std::string& response, const nlohmann::json& request) {
    U32 requested = 0;
    if (request.contains("reasoning")) {
        requested |= kReasoningSetting;
    }
    if (request.contains("include")) {
        requested |= kEncryptedContent;
    }
    if (request.contains("thinking") || request.contains("output_config")) {
        requested |= kThinking;
    }
    if (request.contains("reasoning_effort")) {
        requested |= kReasoningEffort;
    }

    const bool invalidRequest = failure.kind == Provider::Error::Kind::InvalidRequest;
    const bool badRequestStatus = failure.status == 400 || failure.status == 422;
    if (!invalidRequest || !badRequestStatus || !requested) {
        return 0;
    }

    if (const auto param = RejectedParam(response); !param.empty()) {
        return ParamOptions(param, requested);
    }
    const auto message = Lowercase(failure.message);
    U32 mentioned = MessageOptions(message, requested);
    if (requested & kThinking) {
        mentioned |= ThinkingOptions(message);
    }
    return requested & mentioned;
}

class Capabilities {
 public:
    static std::shared_ptr<Capabilities> For(const Config& config, const Provider::Request& request) {
        static std::mutex mutex;
        static std::unordered_map<std::string, std::shared_ptr<Capabilities>> known;
        std::lock_guard lock(mutex);

        const std::string visibility = config.requestReasoning ? "visible" : "hidden";
        const auto key = Detail::Wire::Origin(config) + '\n' + config.path + '\n' + request.model + '\n' +
                         request.reasoningEffort + '\n' + request.reasoningContext + '\n' + visibility;

        auto& entry = known[key];
        if (!entry) {
            entry = std::make_shared<Capabilities>();
        }
        return entry;
    }

    void strip(nlohmann::json& request) const {
        Remove(request, rejected.load());
    }

    bool reject(const Provider::Error& failure, const std::string& response, nlohmann::json& request) {
        const U32 options = Rejected(failure, response, request);
        if (!options) {
            return false;
        }

        Remove(request, options);
        rejected.fetch_or(options);
        return true;
    }

 private:
    std::atomic<U32> rejected{0};
};

nlohmann::json RequestBody(const Config& config, const Provider::Request& request, bool reasoning) {
    if (config.api == Provider::Api::Responses) {
        return Detail::Responses::Encode(config, request, reasoning);
    }
    if (config.api == Provider::Api::Messages) {
        return Detail::Messages::Encode(config, request, reasoning);
    }
    return Detail::ChatCompletions::Encode(config, request, reasoning);
}

std::unique_ptr<Detail::EventStream> ResponseStream(const Config& config) {
    if (config.api == Provider::Api::Responses) {
        return Detail::Responses::Decoder();
    }
    if (config.api == Provider::Api::Messages) {
        return Detail::Messages::Decoder();
    }
    return Detail::ChatCompletions::Decoder();
}

Provider::Error Validate(const Config& config, bool models, const std::string& model, const std::string& key) {
    using Kind = Provider::Error::Kind;

    if (config.host.empty()) {
        return {Kind::Configuration, 0, "No provider endpoint is configured."};
    }
    if (!config.host.starts_with("https://") && !config.host.starts_with("http://")) {
        return {Kind::Configuration, 0, "The base URL must begin with http:// or https://."};
    }

    const auto& path = models ? config.modelsPath : config.path;
    if (path.empty() || path.front() != '/') {
        return {Kind::Configuration, 0, "The API path must be a non-empty path beginning with '/'."};
    }
    if (!models && model.empty()) {
        return {Kind::Configuration, 0, "Select a model before sending a message."};
    }
    if (config.timeoutSeconds == 0) {
        return {Kind::Configuration, 0, "The request timeout must be positive."};
    }

    const auto scope = models ? Provider::Scope::Models : Provider::Scope::Chat;
    if ((config.requireApiKey && key.empty()) ||
        !Provider::Profile::Find(config.kind).ready(scope, !key.empty())) {
        return {Kind::MissingCredentials, 0, Detail::Diagnostics::MissingKey(config.kind)};
    }

    return {};
}

Platform::Http::Headers Headers(const Config& config, const std::string& key, const std::string& sessionId) {
    Platform::Http::Headers headers{{"User-Agent", config.userAgent}};

    const auto& profile = Provider::Profile::Find(config.kind);
    for (const auto& [name, value] : profile.headers) {
        headers.emplace_back(name, value);
    }
    if (!profile.sessionHeader.empty() && !sessionId.empty()) {
        headers.emplace_back(profile.sessionHeader, sessionId);
    }

    if (config.api == Provider::Api::Messages) {
        headers.emplace_back("anthropic-version", Detail::Messages::Version);
        if (!key.empty()) {
            headers.emplace_back("x-api-key", key);
        }
        return headers;
    }

    if (!key.empty()) {
        headers.emplace_back("Authorization", "Bearer " + key);
    }
    return headers;
}

class ChatCall {
 public:
    ChatCall(const Config& config, const Provider::Request& request,
             const Provider::Callbacks& callbacks, Provider::Response& response) :
        config(config),
        request(request),
        capabilities(Capabilities::For(config, request)),
        callbacks(callbacks),
        response(response) {
        if (this->request.sessionId.empty()) {
            this->request.sessionId = Random::HexToken();
        }
    }

    Result run() {
        if (cancelled()) {
            return failCancelled();
        }
        key = ReadKey(config);
        if (auto failure = Validate(config, false, request.model, key)) {
            return fail(failure);
        }

        try {
            body = RequestBody(config, request, static_cast<bool>(callbacks.onReasoning));
            capabilities->strip(body);
            return attempts();
        } catch (const std::exception& exception) {
            return fail({Provider::Error::Kind::Protocol, 0, exception.what()});
        }
    }

 private:
    struct Attempt {
        std::unique_ptr<Detail::EventStream> stream;
        I32 status = 0;
        std::string body;
        std::string retryAfter;

        bool consume(std::string_view bytes) {
            constexpr size_t limit = 65536;
            if (body.size() < limit) {
                body.append(bytes.data(), std::min(bytes.size(), limit - body.size()));
            }

            const bool success = status >= 200 && status < 300;
            if (!success) {
                return true;
            }
            return stream->append(bytes);
        }
    };

    bool cancelled() const {
        return callbacks.shouldAbort && callbacks.shouldAbort();
    }

    Result fail(Provider::Error failure) const {
        return Fail(std::move(failure), &response.error, config, key);
    }

    Result failCancelled() const {
        return fail({Provider::Error::Kind::Cancelled, 0, "Request cancelled."});
    }

    Result attempts() {
        while (true) {
            if (cancelled()) {
                return failCancelled();
            }

            Attempt attempt{ResponseStream(config)};
            attempt.stream->onToken = callbacks.onToken;
            attempt.stream->onReasoning = callbacks.onReasoning;
            attempt.stream->onToolCall = callbacks.onToolCall;

            const auto result = send(attempt);
            if (cancelled()) {
                response.text = attempt.stream->text;
                return failCancelled();
            }

            auto failure = diagnose(attempt, result);
            response.text = attempt.stream->text;
            response.usage = attempt.stream->usage;
            if (!failure) {
                return succeed(attempt);
            }

            if (const auto outcome = recover(std::move(failure), attempt)) {
                return *outcome;
            }
        }
    }

    Platform::Http::Client::Response send(Attempt& attempt) const {
        Platform::Http::Client::Request http;
        http.method = "POST";
        http.host = config.host;
        http.path = config.path;
        http.headers = Headers(config, key, request.sessionId);
        http.headers.emplace_back("Accept", "text/event-stream");
        http.headers.emplace_back("Content-Type", "application/json");
        http.body = body.dump();
        http.timeoutSeconds = config.timeoutSeconds;
        http.cancelled = callbacks.shouldAbort;
        http.onResponse = [&attempt](I32 status, const Platform::Http::Headers&) {
            attempt.status = status;
            return true;
        };
        http.onData = [&attempt](std::string_view data) {
            return attempt.consume(data);
        };

        auto response = Platform::Http::Client::Send(http);
        attempt.retryAfter = response.header("Retry-After");
        return response;
    }

    Provider::Error diagnose(Attempt& attempt, const Platform::Http::Client::Response& result) const {
        auto& stream = *attempt.stream;

        const bool informational = attempt.status != 0 && attempt.status < 200;
        if (attempt.status >= 300 || informational) {
            return Detail::Diagnostics::Http(attempt.status, attempt.body, attempt.retryAfter, config.kind);
        }
        if (stream.error) {
            return stream.error;
        }
        if (result.error) {
            return Detail::Diagnostics::Transport(result.error);
        }
        if (stream.finish()) {
            return {};
        }

        const auto payload = nlohmann::json::parse(attempt.body, nullptr, false);
        if (payload.is_object() && payload.contains("error")) {
            return Detail::Diagnostics::Payload(payload, 0, {}, config.kind);
        }
        return stream.error;
    }

    std::optional<Result> recover(Provider::Error failure, const Attempt& attempt) {
        if (failure.retryAfter.empty()) {
            failure.retryAfter = attempt.retryAfter;
        }
        Detail::Diagnostics::Sanitize(failure, key);

        if (dropRejectedOptions(failure, attempt) || dropUnboundThinking(failure, attempt)) {
            return std::nullopt;
        }

        std::optional<std::chrono::milliseconds> delay;
        if (!attempt.stream->outputStarted) {
            delay = Detail::Retry::Delay(failure, retries, config.retries);
        }
        if (!delay) {
            return fail(std::move(failure));
        }

        JST_WARN("[AGENT_CLIENT] {} Retrying in {} ms (attempt {} of {}).",
                 failure.message, delay->count(), retries + 1, config.retries);
        if (!Detail::Retry::Wait(*delay, callbacks.shouldAbort)) {
            return failCancelled();
        }

        ++retries;
        return std::nullopt;
    }

    bool dropRejectedOptions(const Provider::Error& failure, const Attempt& attempt) {
        if (attempt.stream->outputStarted) {
            return false;
        }

        if (!capabilities->reject(failure, attempt.body, body)) {
            return false;
        }

        JST_WARN("[AGENT_CLIENT] Provider rejected reasoning options ({}). Retrying without them.",
                 failure.message);
        return true;
    }

    bool dropUnboundThinking(const Provider::Error& failure, const Attempt& attempt) {
        const bool rejected = failure.kind == Provider::Error::Kind::InvalidRequest && failure.status == 400;
        if (config.api != Provider::Api::Messages || droppedThinking || !rejected || attempt.stream->outputStarted) {
            return false;
        }
        if (Lowercase(failure.message).find("bound to a different conversation") == std::string::npos) {
            return false;
        }
        if (!Detail::Messages::DropThinking(body)) {
            return false;
        }

        droppedThinking = true;
        JST_WARN("[AGENT_CLIENT] Provider rejected replayed thinking ({}). Retrying without it.", failure.message);
        return true;
    }

    Result succeed(Attempt& attempt) const {
        auto& stream = *attempt.stream;
        response.toolCalls = std::move(stream.toolCalls);
        response.usage = stream.usage;

        auto payload = stream.payload();
        if (!payload.empty()) {
            response.state = std::make_shared<const Provider::State>(Provider::State{
                Detail::Wire::Origin(config),
                config.path,
                request.model,
                std::move(payload),
            });
        }
        return Result::SUCCESS;
    }

    const Config& config;
    Provider::Request request;
    std::shared_ptr<Capabilities> capabilities;
    const Provider::Callbacks& callbacks;
    Provider::Response& response;
    std::string key;
    nlohmann::json body;
    U32 retries = 0;
    bool droppedThinking = false;
};

}  // namespace

Provider::Provider(Config config) : configuration(std::move(config)) {
    if (configuration.userAgent.empty()) {
        configuration.userAgent = std::string("CyberEther/") + JETSTREAM_VERSION_STR;
    }
}

Provider::~Provider() = default;

const Provider::Config& Provider::config() const {
    return configuration;
}

const Provider::Profile& Provider::profile() const {
    return Profile::Find(configuration.kind);
}

Result Provider::models(std::vector<Model>& models, Provider::Error* error,
                        const Cancellation& cancelled) const {
    const auto& config = configuration;
    models.clear();
    if (error) {
        *error = {};
    }

    const auto key = ReadKey(config);
    if (auto failure = Validate(config, true, {}, key)) {
        return Fail(failure, error, config, key, " model discovery");
    }

    try {
        Platform::Http::Client::Request http;
        http.host = config.host;
        http.path = config.modelsPath;
        http.headers = Headers(config, key, {});
        http.timeoutSeconds = std::min<U64>(config.timeoutSeconds, 15);
        http.cancelled = cancelled;

        nlohmann::json body;
        if (auto failure = Detail::Diagnostics::Response(Platform::Http::Client::Send(http), body)) {
            return Fail(std::move(failure), error, config, key, " model discovery");
        }

        if (config.kind == Provider::Kind::ChatGPT) {
            if (!body.is_object() || !body.contains("models") || !body["models"].is_array()) {
                return Fail({Provider::Error::Kind::Protocol, 0, "Expected a JSON object containing a 'models' array."},
                            error, config, key, " model discovery");
            }

            for (const auto& item : body["models"]) {
                if (!item.is_object() || Detail::Wire::Field(item, "visibility") != "list") {
                    continue;
                }
                auto slug = Detail::Wire::Field(item, "slug");
                if (slug.empty()) {
                    continue;
                }
                models.push_back({std::move(slug), Detail::Wire::Field(item, "display_name")});
            }
            return Result::SUCCESS;
        }

        if (!body.is_object() || !body.contains("data") || !body["data"].is_array()) {
            return Fail({Provider::Error::Kind::Protocol, 0, "Expected a JSON object containing a 'data' array."},
                        error, config, key, " model discovery");
        }

        for (const auto& item : body["data"]) {
            if (!item.is_object() || !item.contains("id") || !item["id"].is_string()) {
                continue;
            }

            Model model{item["id"].get<std::string>(), {}};
            if (item.contains("name") && item["name"].is_string()) {
                model.name = item["name"].get<std::string>();
            }
            if (item.contains("context_length") && item["context_length"].is_number_unsigned()) {
                model.contextLength = item["context_length"].get<U64>();
            }

            const bool hasOutputLimit = item.contains("max_output_tokens") &&
                                        item["max_output_tokens"].is_number_unsigned();
            const bool hasTopProvider = item.contains("top_provider") && item["top_provider"].is_object();
            if (hasOutputLimit) {
                model.outputLength = item["max_output_tokens"].get<U64>();
            } else if (hasTopProvider) {
                const auto& topProvider = item["top_provider"];
                const auto limit = topProvider.find("max_completion_tokens");
                if (limit != topProvider.end() && limit->is_number_unsigned()) {
                    model.outputLength = limit->get<U64>();
                }
            }

            models.push_back(std::move(model));
        }

        std::sort(models.begin(), models.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.id < rhs.id;
        });

        return Result::SUCCESS;
    } catch (const std::exception& exception) {
        return Fail({Provider::Error::Kind::Configuration, 0, exception.what()}, error, config, key, " model discovery");
    }
}

Result Provider::stream(const Request& request, const Callbacks& callbacks, Response& response) const {
    response = {};

    return ChatCall(configuration, request, callbacks, response).run();
}

}  // namespace Jetstream::Agent
