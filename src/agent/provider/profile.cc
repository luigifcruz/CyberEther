#include "jetstream/agent/provider.hh"

#include "tools/text.hh"

#include <algorithm>
#include <array>

namespace Jetstream::Agent {

bool Provider::Profile::ready(Scope scope, bool hasKey) const {
    return hasKey || (scope == Scope::Models ? anonymousModels : anonymousChat);
}

std::span<const Provider::Profile> Provider::Profile::All() {
    static const std::array profiles{
        Profile{
            .kind = Kind::OpenAI,
            .name = "OpenAI",
            .description = "OpenAI models through the Responses API.",
            .host = "https://api.openai.com",
            .api = Api::Responses,
            .chatPath = "/v1/responses",
            .livePath = "/v1/live/sessions",
            .catalogId = "openai",
            .requestReasoning = true,
            .promptCacheKey = true,
        },
        Profile{
            .kind = Kind::OpenRouter,
            .name = "OpenRouter",
            .description = "Routes to hundreds of models behind a single API key.",
            .host = "https://openrouter.ai",
            .chatPath = "/api/v1/chat/completions",
            .modelsPath = "/api/v1/models",
            .catalogId = "openrouter",
            .requestReasoning = true,
            .anonymousModels = true,
            .nestedReasoningEffort = true,
            .cacheBreakpointModels = {"anthropic/", "~anthropic/"},
            .sessionHeader = "x-session-id",
            .headers = {
                {"HTTP-Referer", "https://github.com/luigifcruz/CyberEther"},
                {"X-Title", "CyberEther"},
            },
        },
        Profile{
            .kind = Kind::OpenCodeGo,
            .name = "OpenCode Go",
            .description = "Low-cost subscription access to curated coding models.",
            .host = "https://opencode.ai",
            .chatPath = "/zen/go/v1/chat/completions",
            .modelsPath = "/zen/go/v1/models",
            .catalogId = "opencode-go",
            .sessionHeader = "x-opencode-session",
        },
        Profile{
            .kind = Kind::Fireworks,
            .name = "Fireworks AI",
            .description = "Fast open-weight model serving by Fireworks AI.",
            .host = "https://api.fireworks.ai",
            .chatPath = "/inference/v1/chat/completions",
            .modelsPath = "/inference/v1/models",
            .catalogId = "fireworks-ai",
        },
        Profile{
            .kind = Kind::Custom,
            .name = "Custom",
            .description = "Any OpenAI-compatible endpoint, local or remote.",
            .anonymousModels = true,
            .anonymousChat = true,
        },
    };
    return profiles;
}

const Provider::Profile& Provider::Profile::Find(Kind kind) {
    const auto profiles = All();
    const auto found = std::find_if(profiles.begin(), profiles.end(), [kind](const auto& profile) {
        return profile.kind == kind;
    });
    if (found == profiles.end()) {
        return profiles.back();
    }
    return *found;
}

Provider::Kind Provider::Profile::Identify(const std::string& host, const std::string& chatPath) {
    std::string origin = Text::ToLower(host);
    while (!origin.empty() && origin.back() == '/') {
        origin.pop_back();
    }

    for (const auto& profile : All()) {
        const bool sameHost = !profile.host.empty() && origin == profile.host;
        const bool samePath = chatPath.empty() || chatPath == profile.chatPath;
        if (sameHost && samePath) {
            return profile.kind;
        }
    }

    return Kind::Custom;
}

const char* Provider::Name(Api api) {
    return api == Api::Responses ? "responses" : "chat_completions";
}

Provider::Api Provider::Parse(const std::string& name, Api fallback) {
    if (name == Name(Api::Responses)) {
        return Api::Responses;
    }
    if (name == Name(Api::ChatCompletions)) {
        return Api::ChatCompletions;
    }
    return fallback;
}

}  // namespace Jetstream::Agent
