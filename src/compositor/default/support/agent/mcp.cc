#include "mcp.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <unordered_set>

namespace Jetstream {

namespace {

using Json = nlohmann::json;

constexpr const char* kPath = "/mcp";
constexpr U64 kMaxRequestBytes = 8 * 1024 * 1024;
constexpr std::array<const char*, 2> kProtocolVersions = {
    "2025-11-25",
    "2025-06-18",
};

bool IsSupportedVersion(const std::string& version) {
    return std::find(kProtocolVersions.begin(), kProtocolVersions.end(), version) != kProtocolVersions.end();
}

bool IsWildcard(const std::string& address) {
    return address == "0.0.0.0" || address == "::" || address == "[::]";
}

std::string HostForUrl(const std::string& host) {
    if (host.find(':') != std::string::npos && host.front() != '[') {
        return "[" + host + "]";
    }
    return host;
}

bool IsLoopbackOrigin(const std::string& origin) {
    const auto scheme = origin.find("://");
    if (scheme == std::string::npos) {
        return false;
    }

    std::string host = origin.substr(scheme + 3);
    if (host.starts_with("[")) {
        host = host.substr(0, host.find(']') + 1);
    } else {
        host = host.substr(0, host.find_first_of(":/"));
    }

    return host == "localhost" || host == "127.0.0.1" || host == "[::1]";
}

Platform::Http::Server::Response JsonResponse(const Json& body) {
    return {.contentType = "application/json", .body = body.dump()};
}

Json Reply(const Json& id, Json result) {
    return {{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}};
}

Json Failure(const Json& id, int code, const std::string& message) {
    return {{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", message}}}};
}

Json Content(const Agent::Tools::Output& output) {
    Json content = Json::array();
    if (!output.content.empty() || output.images.empty()) {
        content.push_back({{"type", "text"}, {"text", output.content}});
    }

    for (const auto& image : output.images) {
        const auto separator = image.find(";base64,");
        if (!image.starts_with("data:") || separator == std::string::npos) {
            continue;
        }
        content.push_back({
            {"type", "image"},
            {"mimeType", image.substr(5, separator - 5)},
            {"data", image.substr(separator + 8)},
        });
    }

    return content;
}

}  // namespace

struct AgentMcp::Impl {
    Config config;
    Platform::Http::Server server;
    std::shared_ptr<std::atomic<bool>> stopping = std::make_shared<std::atomic<bool>>(false);
    std::unordered_set<std::string> names;
    Json tools = Json::array();
    std::string url;

    Json handle(const Json& message) {
        const Json id = message.contains("id") ? message.at("id") : Json();
        const auto method = message.value("method", std::string{});
        const Json params = message.contains("params") ? message.at("params") : Json::object();

        if (method == "initialize") {
            const auto requested = params.value("protocolVersion", std::string{});
            const std::string version = IsSupportedVersion(requested) ? requested : kProtocolVersions.front();

            return Reply(id, {
                {"protocolVersion", version},
                {"capabilities", {{"tools", {{"listChanged", false}}}}},
                {"serverInfo", {{"name", "CyberEther"}, {"version", JETSTREAM_VERSION_STR}}},
                {"instructions", config.instructions},
            });
        }

        if (method == "ping") {
            return Reply(id, Json::object());
        }

        if (method == "tools/list") {
            return Reply(id, {{"tools", tools}});
        }

        if (method == "tools/call") {
            const auto name = params.value("name", std::string{});
            if (!names.contains(name)) {
                return Failure(id, -32602, "Unknown tool: " + name);
            }

            const Json arguments = params.contains("arguments") ? params.at("arguments") : Json::object();
            if (!arguments.is_object()) {
                return Failure(id, -32602, "Tool arguments must be an object.");
            }

            const auto output = config.executeTool(name, arguments.dump(), [stopping = stopping]() {
                return stopping->load();
            });

            return Reply(id, {
                {"content", Content(output)},
                {"isError", output.error},
            });
        }

        return Failure(id, -32601, "Method not found: " + method);
    }

    Platform::Http::Server::Response serve(const Platform::Http::Server::Request& request) {
        if (request.path != kPath) {
            return {.status = 404};
        }
        if (request.method != "POST") {
            return {.status = 405, .headers = {{"Allow", "POST"}}};
        }

        const auto origin = request.header("Origin");
        if (!origin.empty() && !IsLoopbackOrigin(origin)) {
            return {.status = 403};
        }

        const auto protocol = request.header("MCP-Protocol-Version");
        if (!protocol.empty() && !IsSupportedVersion(protocol)) {
            return {.status = 400, .contentType = "text/plain", .body = "Unsupported MCP protocol version."};
        }

        Json message;
        try {
            message = Json::parse(request.body);
        } catch (const Json::exception&) {
            return JsonResponse(Failure(nullptr, -32700, "Parse error."));
        }

        const auto version = message.is_object() ? message.find("jsonrpc") : message.end();
        if (version == message.end() || *version != "2.0") {
            return JsonResponse(Failure(nullptr, -32600, "Invalid request."));
        }

        if (!message.contains("method") || !message.contains("id")) {
            return {.status = 202};
        }

        try {
            return JsonResponse(handle(message));
        } catch (const Json::exception& e) {
            return JsonResponse(Failure(message.at("id"), -32602, e.what()));
        }
    }
};

Result AgentMcp::start(Config config) {
    stop();

    auto next = std::make_unique<Impl>();
    next->config = std::move(config);

    for (const auto& function : next->config.tools) {
        Json schema = Json::parse(function.parametersJson, nullptr, false);
        if (schema.is_discarded() || !schema.is_object()) {
            schema = {{"type", "object"}};
        }
        next->tools.push_back({
            {"name", function.name},
            {"description", function.description},
            {"inputSchema", std::move(schema)},
        });
        next->names.insert(function.name);
    }

    const auto& address = next->config.address;
    const auto status = next->server.start({
        .address = address,
        .port = next->config.port,
        .maxBodyBytes = kMaxRequestBytes,
        .handler = [impl = next.get()](const Platform::Http::Server::Request& request) {
            return impl->serve(request);
        },
    });
    if (status != Result::SUCCESS) {
        JST_ERROR("[AGENT_MCP] Failed to start the MCP server on {}:{}.", address, next->config.port);
        return Result::ERROR;
    }

    std::string host = address;
#ifndef JST_OS_BROWSER
    if (IsWildcard(host) && Platform::LocalNetworkAddress(host) != Result::SUCCESS) {
        host = "127.0.0.1";
    }
#endif
    next->url = jst::fmt::format("http://{}:{}{}", HostForUrl(host), next->server.port(), kPath);

    JST_INFO("[AGENT_MCP] MCP server listening on {}.", next->url);
    if (IsWildcard(address)) {
        JST_WARN("[AGENT_MCP] The MCP server accepts connections from every interface.");
    }

    impl = std::move(next);
    return Result::SUCCESS;
}

void AgentMcp::stop() {
    if (!impl) {
        return;
    }

    impl->stopping->store(true);
    impl->server.stop();
    impl.reset();

    JST_INFO("[AGENT_MCP] MCP server stopped.");
}

AgentMcp::AgentMcp() = default;

AgentMcp::~AgentMcp() {
    stop();
}

bool AgentMcp::running() const {
    return impl != nullptr;
}

const std::string& AgentMcp::url() const {
    static const std::string empty;
    return impl ? impl->url : empty;
}

}  // namespace Jetstream
