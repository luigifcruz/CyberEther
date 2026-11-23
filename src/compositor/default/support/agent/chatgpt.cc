#include "chatgpt.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include "tools/base64.hh"

#include <nlohmann/json.hpp>

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE) && defined(JETSTREAM_PLATFORM_HTTP_AVAILABLE) && \
    !defined(JST_OS_BROWSER) && !defined(JST_OS_IOS)
#define JETSTREAM_AGENT_CHATGPT_AVAILABLE
#endif

#if defined(JETSTREAM_AGENT_CHATGPT_AVAILABLE)
#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/evp.h>
#include <openssl/param_build.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace Jetstream {

namespace {

using Json = nlohmann::json;

constexpr const char* kAuthHost = "https://auth.openai.com";
constexpr const char* kIssuer = "https://auth.openai.com";
constexpr const char* kAuthorizePath = "/api/accounts/authorize";
constexpr const char* kTokenPath = "/api/accounts/oauth/token";
constexpr const char* kDiscoveryPath = "/.well-known/openid-configuration";
constexpr const char* kResource = "https://api.openai.com/v1";
constexpr const char* kScope = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
constexpr const char* kPlanScope = "chatgpt.tokens.use.direct";
constexpr const char* kRegistrationClient = "dynamic_agent_client";
constexpr const char* kAgentName = "CyberEther";
constexpr const char* kCallbackHost = "127.0.0.1";
constexpr const char* kCallbackPath = "/auth/callback";
constexpr U16 kPreferredPort = 1455;
constexpr auto kSignInTimeout = std::chrono::minutes(10);
constexpr I64 kRefreshMargin = 120;
constexpr I64 kClockSkew = 300;

constexpr std::array<std::string_view, 6> kTerminalRefreshErrors{
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
};

std::string PercentEncode(std::string_view value) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size());
    for (const unsigned char character : value) {
        const bool unreserved = (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') ||
                                (character >= '0' && character <= '9') || character == '-' || character == '.' ||
                                character == '_' || character == '~';
        if (unreserved) {
            out += static_cast<char>(character);
            continue;
        }
        out += '%';
        out += kHex[character >> 4];
        out += kHex[character & 0x0F];
    }
    return out;
}

std::string PercentDecode(std::string_view value) {
    const auto hex = [](char character) -> int {
        if (character >= '0' && character <= '9') {
            return character - '0';
        }
        if (character >= 'a' && character <= 'f') {
            return character - 'a' + 10;
        }
        if (character >= 'A' && character <= 'F') {
            return character - 'A' + 10;
        }
        return -1;
    };

    std::string out;
    out.reserve(value.size());
    for (U64 index = 0; index < value.size(); ++index) {
        const char character = value[index];
        if (character == '+') {
            out += ' ';
            continue;
        }
        if (character == '%' && index + 2 < value.size()) {
            const int high = hex(value[index + 1]);
            const int low = hex(value[index + 2]);
            if (high >= 0 && low >= 0) {
                out += static_cast<char>((high << 4) | low);
                index += 2;
                continue;
            }
        }
        out += character;
    }
    return out;
}

std::string FormBody(const std::vector<std::pair<std::string, std::string>>& fields) {
    std::string body;
    for (const auto& [name, value] : fields) {
        if (!body.empty()) {
            body += '&';
        }
        body += PercentEncode(name) + "=" + PercentEncode(value);
    }
    return body;
}

std::string Field(const Json& object, const char* name) {
    const auto found = object.find(name);
    return found != object.end() && found->is_string() ? found->get<std::string>() : std::string{};
}

bool HasScope(const std::string& scopes, std::string_view scope) {
    std::istringstream stream(scopes);
    std::string entry;
    while (stream >> entry) {
        if (entry == scope) {
            return true;
        }
    }
    return false;
}

#if defined(JETSTREAM_AGENT_CHATGPT_AVAILABLE)

std::string Base64Url(const U8* data, U64 size) {
    auto encoded = Base64::Encode(data, size);
    for (auto& character : encoded) {
        if (character == '+') {
            character = '-';
        } else if (character == '/') {
            character = '_';
        }
    }
    while (!encoded.empty() && encoded.back() == '=') {
        encoded.pop_back();
    }
    return encoded;
}

std::vector<U8> Base64UrlDecode(std::string value) {
    for (auto& character : value) {
        if (character == '-') {
            character = '+';
        } else if (character == '_') {
            character = '/';
        }
    }
    while (value.size() % 4 != 0) {
        value += '=';
    }
    return Base64::Decode(value);
}

std::pair<std::string, std::string> SplitUrl(const std::string& url) {
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) {
        return {};
    }
    const auto path = url.find('/', scheme + 3);
    if (path == std::string::npos) {
        return {url, "/"};
    }
    return {url.substr(0, path), url.substr(path)};
}

std::string Html(const std::string& title, const std::string& message) {
    return "<!doctype html><html><head><meta charset=\"utf-8\"><title>" + title +
           "</title><style>body{font-family:system-ui,sans-serif;background:#0b0d12;color:#e6e8ee;"
           "display:flex;align-items:center;justify-content:center;height:100vh;margin:0}"
           "main{text-align:center;max-width:28rem;padding:1rem}h1{font-size:1.25rem}"
           "p{color:#9aa3b2}</style></head><body><main><h1>" + title + "</h1><p>" + message +
           "</p></main></body></html>";
}

std::string RandomToken(U64 size) {
    std::vector<U8> bytes(size);
    if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
        return {};
    }
    return Base64Url(bytes.data(), bytes.size());
}

struct Reply {
    I32 status = 0;
    Json body;
    std::string error;
    std::string description;
    std::string transport;

    bool ok() const {
        return transport.empty() && status >= 200 && status < 300;
    }

    std::string message(const std::string& context) const {
        if (!transport.empty()) {
            return context + ": " + transport + ".";
        }
        std::string detail = description.empty() ? error : description;
        if (detail.empty()) {
            detail = Field(body, "detail");
        }
        if (detail.empty()) {
            return jst::fmt::format("{} (HTTP {}).", context, status);
        }
        if (detail.ends_with('.')) {
            detail.pop_back();
        }
        return jst::fmt::format("{}: {} (HTTP {}).", context, detail, status);
    }
};

Reply Send(const std::string& method, const std::string& host, const std::string& path, const std::string& form,
           const Agent::Cancellation& cancelled = {}) {
    Platform::Http::Client::Request request;
    request.method = method;
    request.host = host;
    request.path = path;
    request.headers = {{"Accept", "application/json"}, {"User-Agent", kAgentName}};
    if (method == "POST") {
        request.headers.emplace_back("Content-Type", "application/x-www-form-urlencoded");
        request.body = form;
    }
    request.timeoutSeconds = 30;
    request.cancelled = cancelled;

    const auto response = Platform::Http::Client::Send(request);
    Reply reply;
    reply.status = response.status;
    if (response.error) {
        reply.transport = response.error.message.empty() ? "Network request failed" : response.error.message;
        return reply;
    }
    reply.body = Json::parse(response.body, nullptr, false);
    if (reply.body.is_discarded()) {
        reply.body = Json::object();
    }
    if (reply.body.is_object()) {
        const auto error = reply.body.find("error");
        if (error != reply.body.end() && error->is_string()) {
            reply.error = error->get<std::string>();
        } else if (error != reply.body.end() && error->is_object()) {
            reply.error = Field(*error, "code");
            reply.description = Field(*error, "message");
        }
        if (reply.description.empty()) {
            reply.description = Field(reply.body, "error_description");
        }
    }
    return reply;
}

class Jwks {
 public:
    static Jwks& Shared() {
        static Jwks instance;
        return instance;
    }

    std::string endpoint(const char* name, std::string& error) {
        std::lock_guard lock(mutex);
        if (!loadDiscovery(error)) {
            return {};
        }
        return Field(discovery, name);
    }

    std::optional<Json> key(const std::string& kid, std::string& error) {
        std::lock_guard lock(mutex);
        if (auto found = find(kid)) {
            return found;
        }
        if (!loadKeys(error)) {
            return std::nullopt;
        }
        if (auto found = find(kid)) {
            return found;
        }
        error = "The sign-in token uses an unknown signing key";
        return std::nullopt;
    }

 private:
    bool loadDiscovery(std::string& error) {
        if (discovery.is_object()) {
            return true;
        }
        const auto reply = Send("GET", kAuthHost, kDiscoveryPath, {});
        if (!reply.ok() || !reply.body.is_object()) {
            error = reply.message("Can't load the OpenAI sign-in configuration");
            return false;
        }
        discovery = reply.body;
        return true;
    }

    bool loadKeys(std::string& error) {
        if (!loadDiscovery(error)) {
            return false;
        }
        const auto [host, path] = SplitUrl(Field(discovery, "jwks_uri"));
        if (host.empty() || !host.starts_with("https://")) {
            error = "The OpenAI sign-in configuration has no signing keys";
            return false;
        }
        const auto reply = Send("GET", host, path, {});
        if (!reply.ok() || !reply.body.is_object() || !reply.body.contains("keys")) {
            error = reply.message("Can't load the OpenAI signing keys");
            return false;
        }
        keys = reply.body["keys"];
        return true;
    }

    std::optional<Json> find(const std::string& kid) const {
        if (!keys.is_array()) {
            return std::nullopt;
        }
        for (const auto& entry : keys) {
            if (entry.is_object() && Field(entry, "kid") == kid && Field(entry, "kty") == "RSA") {
                return entry;
            }
        }
        return std::nullopt;
    }

    std::mutex mutex;
    Json discovery;
    Json keys;
};

bool VerifyRs256(const Json& jwk, std::string_view message, const std::vector<U8>& signature) {
    const auto modulus = Base64UrlDecode(Field(jwk, "n"));
    const auto exponent = Base64UrlDecode(Field(jwk, "e"));
    if (modulus.empty() || exponent.empty() || signature.empty()) {
        return false;
    }

    std::unique_ptr<BIGNUM, decltype(&BN_free)> n(BN_bin2bn(modulus.data(), static_cast<int>(modulus.size()),
                                                             nullptr), BN_free);
    std::unique_ptr<BIGNUM, decltype(&BN_free)> e(BN_bin2bn(exponent.data(), static_cast<int>(exponent.size()),
                                                             nullptr), BN_free);
    std::unique_ptr<OSSL_PARAM_BLD, decltype(&OSSL_PARAM_BLD_free)> builder(OSSL_PARAM_BLD_new(),
                                                                             OSSL_PARAM_BLD_free);
    if (!n || !e || !builder || OSSL_PARAM_BLD_push_BN(builder.get(), OSSL_PKEY_PARAM_RSA_N, n.get()) != 1 ||
        OSSL_PARAM_BLD_push_BN(builder.get(), OSSL_PKEY_PARAM_RSA_E, e.get()) != 1) {
        return false;
    }
    std::unique_ptr<OSSL_PARAM, decltype(&OSSL_PARAM_free)> params(OSSL_PARAM_BLD_to_param(builder.get()),
                                                                    OSSL_PARAM_free);
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> context(
        EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr), EVP_PKEY_CTX_free);
    if (!params || !context || EVP_PKEY_fromdata_init(context.get()) != 1) {
        return false;
    }
    EVP_PKEY* raw = nullptr;
    if (EVP_PKEY_fromdata(context.get(), &raw, EVP_PKEY_PUBLIC_KEY, params.get()) != 1 || !raw) {
        return false;
    }
    std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(raw, EVP_PKEY_free);

    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!digest || EVP_DigestVerifyInit(digest.get(), nullptr, EVP_sha256(), nullptr, key.get()) != 1) {
        return false;
    }
    return EVP_DigestVerify(digest.get(), signature.data(), signature.size(),
                            reinterpret_cast<const unsigned char*>(message.data()), message.size()) == 1;
}

bool AudienceMatches(const Json& claims, const std::string& clientId) {
    const auto audience = claims.find("aud");
    if (audience == claims.end()) {
        return false;
    }
    if (audience->is_string()) {
        return audience->get<std::string>() == clientId;
    }
    if (audience->is_array()) {
        return std::any_of(audience->begin(), audience->end(), [&](const auto& entry) {
            return entry.is_string() && entry.template get<std::string>() == clientId;
        });
    }
    return false;
}

bool ValidateIdToken(const std::string& token, const std::string& clientId, const std::string& nonce,
                     AgentChatGpt::Credentials& credentials, std::string& error) {
    const auto first = token.find('.');
    const auto second = first == std::string::npos ? std::string::npos : token.find('.', first + 1);
    if (second == std::string::npos) {
        error = "The sign-in returned a malformed identity token";
        return false;
    }

    const auto headerBytes = Base64UrlDecode(token.substr(0, first));
    const auto claimBytes = Base64UrlDecode(token.substr(first + 1, second - first - 1));
    const auto header = Json::parse(headerBytes.begin(), headerBytes.end(), nullptr, false);
    const auto claims = Json::parse(claimBytes.begin(), claimBytes.end(), nullptr, false);
    if (!header.is_object() || !claims.is_object()) {
        error = "The sign-in returned a malformed identity token";
        return false;
    }
    if (Field(header, "alg") != "RS256") {
        error = "The sign-in identity token uses an unsupported signature";
        return false;
    }

    const auto jwk = Jwks::Shared().key(Field(header, "kid"), error);
    if (!jwk) {
        return false;
    }
    const auto signature = Base64UrlDecode(token.substr(second + 1));
    if (!VerifyRs256(*jwk, std::string_view(token).substr(0, second), signature)) {
        error = "The sign-in identity token has an invalid signature";
        return false;
    }

    const auto expiry = claims.find("exp");
    const bool expired = expiry == claims.end() || !expiry->is_number() ||
                         expiry->get<I64>() + kClockSkew < AgentChatGpt::Now();
    if (Field(claims, "iss") != kIssuer || !AudienceMatches(claims, clientId) || expired) {
        error = "The sign-in identity token is not valid for CyberEther";
        return false;
    }
    if (Field(claims, "nonce") != nonce) {
        error = "The sign-in identity token does not match this sign-in";
        return false;
    }

    const auto subject = Field(claims, "sub");
    if (subject.empty()) {
        error = "The sign-in identity token has no account";
        return false;
    }
    credentials.subject = subject;
    if (const auto email = Field(claims, "email"); !email.empty()) {
        credentials.email = email;
    }
    return true;
}

void ApplyTokens(const Json& body, AgentChatGpt::Credentials& credentials) {
    credentials.accessToken = Field(body, "access_token");
    if (const auto refresh = Field(body, "refresh_token"); !refresh.empty()) {
        credentials.refreshToken = refresh;
    }
    if (const auto scope = Field(body, "scope"); !scope.empty()) {
        credentials.scope = scope;
    }
    I64 lifetime = 3600;
    const auto expires = body.find("expires_in");
    if (expires != body.end() && expires->is_number()) {
        lifetime = expires->get<I64>();
    }
    credentials.expiresAt = AgentChatGpt::Now() + lifetime;
}

struct Callback {
    std::mutex mutex;
    std::condition_variable ready;
    std::string state;
    std::optional<std::map<std::string, std::string>> params;
};

Platform::Http::Server::Response Page(I32 status, const std::string& title, const std::string& message) {
    return {
        .status = status,
        .contentType = "text/html; charset=utf-8",
        .body = Html(title, message),
    };
}

Platform::Http::Server::Response Serve(Callback& callback, const Platform::Http::Server::Request& request) {
    if (request.method != "GET" || request.path != kCallbackPath) {
        return Page(404, "Not Found", "This address is only used to finish signing in to CyberEther.");
    }

    auto params = AgentChatGpt::ParseQuery(request.query);
    std::lock_guard lock(callback.mutex);
    if (callback.params || params["state"] != callback.state) {
        return Page(400, "Sign-in Not Recognized", "Start the sign-in again from CyberEther.");
    }

    const bool denied = params.contains("error");
    callback.params = std::move(params);
    callback.ready.notify_all();
    if (denied) {
        return Page(200, "Sign-in Cancelled", "You can close this tab and return to CyberEther.");
    }
    return Page(200, "Signed In", "You can close this tab and return to CyberEther.");
}

#endif

}  // namespace

bool AgentChatGpt::Credentials::planEnabled() const {
    return HasScope(scope, kPlanScope);
}

bool AgentChatGpt::Credentials::expiring(I64 now) const {
    return accessToken.empty() || expiresAt - kRefreshMargin <= now;
}

std::string AgentChatGpt::Credentials::encode() const {
    return Json{
        {"client_id", clientId},
        {"access_token", accessToken},
        {"refresh_token", refreshToken},
        {"scope", scope},
        {"sub", subject},
        {"email", email},
        {"expires_at", expiresAt},
    }.dump();
}

std::optional<AgentChatGpt::Credentials> AgentChatGpt::Credentials::Decode(const std::string& value) {
    const auto json = Json::parse(value, nullptr, false);
    if (!json.is_object()) {
        return std::nullopt;
    }

    Credentials credentials{
        .clientId = Field(json, "client_id"),
        .accessToken = Field(json, "access_token"),
        .refreshToken = Field(json, "refresh_token"),
        .scope = Field(json, "scope"),
        .subject = Field(json, "sub"),
        .email = Field(json, "email"),
    };
    const auto expires = json.find("expires_at");
    if (expires != json.end() && expires->is_number()) {
        credentials.expiresAt = expires->get<I64>();
    }
    if (credentials.clientId.empty() || credentials.refreshToken.empty()) {
        return std::nullopt;
    }
    return credentials;
}

I64 AgentChatGpt::Now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string AgentChatGpt::AuthorizeUrl(const Authorization& authorization) {
    std::vector<std::pair<std::string, std::string>> fields{
        {"client_id", kRegistrationClient},
        {"agent_name_hint", kAgentName},
        {"ext_agent_host_id", authorization.hostId},
        {"response_type", "code"},
        {"redirect_uri", authorization.redirectUri},
        {"scope", kScope},
        {"resource", kResource},
        {"state", authorization.state},
        {"nonce", authorization.nonce},
        {"code_challenge", authorization.challenge},
        {"code_challenge_method", "S256"},
    };
    return std::string(kAuthHost) + kAuthorizePath + "?" + FormBody(fields);
}

std::map<std::string, std::string> AgentChatGpt::ParseQuery(const std::string& query) {
    std::map<std::string, std::string> params;
    U64 start = 0;
    while (start <= query.size()) {
        auto end = query.find('&', start);
        if (end == std::string::npos) {
            end = query.size();
        }
        const auto pair = std::string_view(query).substr(start, end - start);
        if (!pair.empty()) {
            const auto equals = pair.find('=');
            const auto name = PercentDecode(pair.substr(0, equals));
            const auto value = equals == std::string_view::npos ? std::string{} : PercentDecode(pair.substr(equals + 1));
            params.try_emplace(name, value);
        }
        start = end + 1;
    }
    return params;
}

#if defined(JETSTREAM_AGENT_CHATGPT_AVAILABLE)

bool AgentChatGpt::Available() {
    return true;
}

std::string AgentChatGpt::NewHostId() {
    std::array<U8, 16> bytes{};
    if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
        return {};
    }
    bytes[6] = static_cast<U8>((bytes[6] & 0x0F) | 0x40);
    bytes[8] = static_cast<U8>((bytes[8] & 0x3F) | 0x80);

    std::string hex;
    for (U64 index = 0; index < bytes.size(); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) {
            hex += '-';
        }
        hex += jst::fmt::format("{:02x}", bytes[index]);
    }
    return "urn:uuid:" + hex;
}

std::string AgentChatGpt::Challenge(const std::string& verifier) {
    std::array<U8, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(verifier.data()), verifier.size(), digest.data());
    return Base64Url(digest.data(), digest.size());
}

AgentChatGpt::SignInResult AgentChatGpt::SignIn(const std::string& hostId, const Opener& open,
                                                const Agent::Cancellation& cancelled) {
    const auto fail = [](std::string message) {
        JST_WARN("[AGENT] ChatGPT sign-in failed: {}", message);
        return SignInResult{.success = false, .message = std::move(message)};
    };

    if (hostId.empty()) {
        return fail("Can't create an identifier for this computer.");
    }

    const auto verifier = RandomToken(48);
    Authorization authorization{
        .hostId = hostId,
        .state = RandomToken(24),
        .nonce = RandomToken(24),
        .challenge = Challenge(verifier),
    };
    if (verifier.empty() || authorization.state.empty() || authorization.nonce.empty()) {
        return fail("Can't generate a secure sign-in request.");
    }

    auto callback = std::make_shared<Callback>();
    callback->state = authorization.state;

    Platform::Http::Server server;
    const auto handler = [callback](const Platform::Http::Server::Request& request) {
        return Serve(*callback, request);
    };
    if (server.start({.address = kCallbackHost, .port = kPreferredPort, .maxBodyBytes = 4096, .handler = handler}) !=
            Result::SUCCESS &&
        server.start({.address = kCallbackHost, .port = 0, .maxBodyBytes = 4096, .handler = handler}) !=
            Result::SUCCESS) {
        return fail("Can't listen for the sign-in response on this computer.");
    }
    authorization.redirectUri = jst::fmt::format("http://{}:{}{}", kCallbackHost, server.port(), kCallbackPath);

    open(AuthorizeUrl(authorization));

    std::map<std::string, std::string> params;
    {
        const auto deadline = std::chrono::steady_clock::now() + kSignInTimeout;
        std::unique_lock lock(callback->mutex);
        while (!callback->params) {
            if (cancelled && cancelled()) {
                lock.unlock();
                server.stop();
                return {.success = false, .message = "Sign-in cancelled."};
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                lock.unlock();
                server.stop();
                return fail("Sign-in timed out. Try again.");
            }
            callback->ready.wait_for(lock, std::chrono::milliseconds(200));
        }
        params = *callback->params;
    }
    server.stop();

    if (params.contains("error")) {
        if (params["error"] == "access_denied") {
            return {.success = false, .message = "Sign-in was cancelled in the browser."};
        }
        const auto description = params["error_description"];
        return fail(description.empty() ? "ChatGPT rejected the sign-in (" + params["error"] + ")."
                                        : description);
    }

    Credentials credentials;
    credentials.clientId = params["client_id"];
    if (credentials.clientId.empty() || credentials.clientId == kRegistrationClient) {
        return fail("ChatGPT did not register CyberEther for this account.");
    }
    if (params["code"].empty()) {
        return fail("ChatGPT did not return an authorization code.");
    }

    const auto reply = Send("POST", kAuthHost, kTokenPath, FormBody({
        {"grant_type", "authorization_code"},
        {"client_id", credentials.clientId},
        {"code", params["code"]},
        {"code_verifier", verifier},
        {"redirect_uri", authorization.redirectUri},
        {"resource", kResource},
    }), cancelled);
    if (!reply.ok()) {
        return fail(reply.message("Can't finish signing in"));
    }

    ApplyTokens(reply.body, credentials);
    if (credentials.accessToken.empty() || credentials.refreshToken.empty()) {
        return fail("ChatGPT did not return the expected tokens.");
    }

    std::string error;
    if (!ValidateIdToken(Field(reply.body, "id_token"), credentials.clientId, authorization.nonce, credentials,
                         error)) {
        return fail(error + ".");
    }
    if (!credentials.planEnabled()) {
        return fail("This sign-in did not allow ChatGPT plan usage. Sign in again and approve plan usage.");
    }

    JST_INFO("[AGENT] Signed in to ChatGPT.");
    return {.success = true, .credentials = std::move(credentials)};
}

AgentChatGpt::RefreshResult AgentChatGpt::Refresh(Credentials& credentials) {
    const auto reply = Send("POST", kAuthHost, kTokenPath, FormBody({
        {"grant_type", "refresh_token"},
        {"client_id", credentials.clientId},
        {"refresh_token", credentials.refreshToken},
        {"resource", kResource},
    }));
    if (!reply.ok()) {
        const bool terminal = reply.error == "invalid_client" ||
                              std::find(kTerminalRefreshErrors.begin(), kTerminalRefreshErrors.end(), reply.error) !=
                                  kTerminalRefreshErrors.end();
        return {.success = false, .terminal = terminal, .message = reply.message("Can't refresh the ChatGPT sign-in")};
    }

    auto next = credentials;
    ApplyTokens(reply.body, next);
    if (next.accessToken.empty()) {
        return {.success = false, .message = "The ChatGPT sign-in refresh returned no access token."};
    }

    credentials = std::move(next);
    return {.success = true};
}

bool AgentChatGpt::Revoke(const Credentials& credentials) {
    std::string error;
    const auto [host, path] = SplitUrl(Jwks::Shared().endpoint("revocation_endpoint", error));
    if (host.empty() || !host.starts_with("https://")) {
        JST_WARN("[AGENT] Can't revoke the ChatGPT sign-in: {}", error.empty() ? "no revocation endpoint" : error);
        return false;
    }

    const auto reply = Send("POST", host, path, FormBody({
        {"token", credentials.refreshToken},
        {"token_type_hint", "refresh_token"},
        {"client_id", credentials.clientId},
    }));
    if (!reply.ok()) {
        JST_WARN("[AGENT] {}", reply.message("Can't revoke the ChatGPT sign-in"));
        return false;
    }
    return true;
}

#else

bool AgentChatGpt::Available() {
    return false;
}

std::string AgentChatGpt::NewHostId() {
    return {};
}

std::string AgentChatGpt::Challenge(const std::string&) {
    return {};
}

AgentChatGpt::SignInResult AgentChatGpt::SignIn(const std::string&, const Opener&, const Agent::Cancellation&) {
    return {.success = false, .message = "Sign in with ChatGPT is unavailable in this build."};
}

AgentChatGpt::RefreshResult AgentChatGpt::Refresh(Credentials&) {
    return {.success = false, .message = "Sign in with ChatGPT is unavailable in this build."};
}

bool AgentChatGpt::Revoke(const Credentials&) {
    return false;
}

#endif

}  // namespace Jetstream
