#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_CHATGPT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_CHATGPT_HH

#include "jetstream/agent/provider.hh"
#include "jetstream/types.hh"

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace Jetstream {

class AgentChatGpt {
 public:
    static constexpr const char* Auth = "chatgpt";
    static constexpr const char* UsageUrl = "https://chatgpt.com/settings/usage";

    struct Credentials {
        std::string clientId;
        std::string accessToken;
        std::string refreshToken;
        std::string scope;
        std::string subject;
        std::string email;
        I64 expiresAt = 0;

        bool planEnabled() const;
        bool expiring(I64 now) const;
        std::string encode() const;
        static std::optional<Credentials> Decode(const std::string& value);
    };

    struct Authorization {
        std::string hostId;
        std::string redirectUri;
        std::string state;
        std::string nonce;
        std::string challenge;
    };

    struct SignInResult {
        bool success = false;
        std::string message;
        std::optional<Credentials> credentials;
    };

    struct RefreshResult {
        bool success = false;
        bool terminal = false;
        std::string message;
    };

    static bool Available();
    static I64 Now();
    static std::string NewHostId();

    using Opener = std::function<void(const std::string& url)>;

    static SignInResult SignIn(const std::string& hostId, const Opener& open, const Agent::Cancellation& cancelled);
    static RefreshResult Refresh(Credentials& credentials);
    static bool Revoke(const Credentials& credentials);

    static std::string Challenge(const std::string& verifier);
    static std::string AuthorizeUrl(const Authorization& authorization);
    static std::map<std::string, std::string> ParseQuery(const std::string& query);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_AGENT_CHATGPT_HH
