#include <catch2/catch_test_macros.hpp>

#include "jetstream/agent/session.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "tools/random.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using namespace Jetstream;
using namespace std::chrono_literals;

namespace {

using Kind = Agent::Session::Segment::Kind;
using Handler = std::function<Result(const Agent::Provider::Request&,
                                     const Agent::Provider::Callbacks&,
                                     Agent::Provider::Response&)>;

constexpr const char* kSecret = "sk-transcript-secret-key";
constexpr const char* kProviderName = "Nemotron Local";
constexpr const char* kModel = "nvidia/nemotron-3-nano:30b";

// A 640x480 PNG header and a 320x240 JPEG header; neither carries pixels.
const std::string kPng = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAIAAAAAAAAA";
const std::string kJpeg = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wAFAAEC/8AAEQgA8AFAAwEiAAIRAQMRAf/Z";
const std::string kOpaque = "data:image/png;base64,dGVzdA==";

struct Client final : Agent::Provider {
    Handler handler;

    explicit Client(Handler handler)
        : Provider({
              .name = kProviderName,
              .host = "https://user:hunter2@inference.example.test",
              .path = "/v1/chat/completions",
              .apiKey = [] {
                  return std::string(kSecret);
              },
          }),
          handler(std::move(handler)) {}

    Result stream(const Request& request, const Callbacks& callbacks, Response& response) const override {
        return handler(request, callbacks, response);
    }
};

struct Sandbox {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("cyberether-transcript-" + Random::HexToken());
    std::filesystem::path folder = root / "nested" / "transcripts";

    Sandbox() {
        std::filesystem::create_directories(root);
    }

    ~Sandbox() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    std::vector<std::filesystem::path> files() const {
        std::vector<std::filesystem::path> found;
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(folder, error)) {
            found.push_back(entry.path());
        }
        std::sort(found.begin(), found.end());
        return found;
    }
};

Agent::Session::Config Config(const Sandbox& sandbox, Handler handler) {
    Agent::Session::Config config;
    config.provider = std::make_shared<Client>(std::move(handler));
    config.model = kModel;
    config.reasoningEffort = "high";
    config.systemContext = "System instructions stay out of the transcript.";
    config.tools = {{"inspect", "Inspect graph", R"({"type":"object"})"}};
    config.toolResultImages = true;
    config.transcript = {true, Platform::PathToUtf8(sandbox.folder)};
    return config;
}

bool Wait(Agent::Session& session) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (session.snapshot().streaming && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    return !session.snapshot().streaming;
}

std::string Read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

bool Contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// Walks the file the way a Markdown reader does and returns the headings that sit outside
// code fences. `closed` is false when a fence is still open at the end of the file.
struct Outline {
    std::vector<std::string> headings;
    bool closed = true;
};

Outline Parse(const std::string& text) {
    Outline outline;
    U64 open = 0;

    std::string::size_type start = 0;
    while (start < text.size()) {
        auto end = text.find('\n', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const auto line = text.substr(start, end - start);
        start = end + 1;

        const auto ticks = std::min(line.find_first_not_of('`'), line.size());
        if (open == 0) {
            if (ticks >= 3) {
                open = ticks;
            } else if (line.starts_with("## ")) {
                outline.headings.push_back(line.substr(3));
            }
        } else if (ticks >= open && ticks == line.size()) {
            open = 0;
        }
    }

    outline.closed = open == 0;
    return outline;
}

std::vector<std::string> Titles(const Outline& outline) {
    static const std::regex stamped(R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z (.+)$)");

    std::vector<std::string> titles;
    for (const auto& heading : outline.headings) {
        std::smatch match;
        REQUIRE(std::regex_match(heading, match, stamped));
        titles.push_back(match[1]);
    }
    return titles;
}

std::string Name(const std::filesystem::path& path) {
    return Platform::PathToUtf8(path.filename());
}

U64 Count(const std::string& text, const std::string& needle) {
    U64 count = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) {
        ++count;
    }
    return count;
}

// Runs one scripted turn and returns the only transcript it left behind.
std::string Chat(const Sandbox& sandbox, Agent::Session::Config config, const std::string& prompt = "Go.") {
    Agent::Session session;
    REQUIRE(session.submit(prompt, config));
    REQUIRE(Wait(session));

    const auto files = sandbox.files();
    REQUIRE(files.size() == 1);
    return Read(files.front());
}

}  // namespace

TEST_CASE("Chat transcripts record every finalized step of a conversation in order", "[agent][transcript]") {
    Sandbox sandbox;

    const std::string thought = "The user wants the FM graph checked.\n"
                                "## A heading the model wrote\n"
                                "```json\n{\"plan\": \"inspect\"}\n"
                                "The fence above is never closed, and ````four```` backticks follow.\n";
    const std::string tail = "[end of the long reasoning]";
    const std::string longThought = std::string(60000, 'r') + tail;
    const std::string oversized = std::string(20000, 'x') + "<end of result>";
    const std::string inspectArgs = "{\"block\":\"fm\",  \"note\":\"keep ```this``` spacing\"}";

    std::atomic<int> step{0};
    std::string midway;

    auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
        switch (step++) {
            case 0:
                callbacks.onReasoning("The user wants the FM graph checked.\n## A heading the model wrote\n", false);
                callbacks.onReasoning("```json\n{\"plan\": \"inspect\"}\n"
                                      "The fence above is never closed, and ````four```` backticks follow.\n", false);
                callbacks.onToken("Checking the ");
                callbacks.onToken("graph ```now```.");
                callbacks.onToolCall("inspect");
                response.text = "Checking the graph ```now```.";
                response.toolCalls = {
                    {"call-1", "inspect", inspectArgs},
                    {"call-2", "capture", "{}"},
                    {"call-3", "dump", R"({"all":true})"},
                    {"call-4", "missing", ""},
                };
                response.usage = {.inputTokens = 1200, .outputTokens = 80, .cachedTokens = 1000, .reasoningTokens = 40};
                return Result::SUCCESS;
            case 1:
                // Everything before this round is already on disk while the turn is still running.
                midway = Read(sandbox.files().front());
                callbacks.onReasoning(longThought, false);
                callbacks.onToken("The FM graph is healthy.");
                response.text = "The FM graph is healthy.";
                response.usage = {.inputTokens = 30000, .outputTokens = 20000};
                return Result::SUCCESS;
            default:
                callbacks.onToken("It is still healthy.");
                response.text = "It is still healthy.";
                return Result::SUCCESS;
        }
    });
    config.executeTool = [&](const std::string& name, const std::string&, const Agent::Cancellation&) {
        if (name == "inspect") {
            return Agent::Tools::Output{"state:\n````\nrunning\n````\n"};
        }
        if (name == "capture") {
            return Agent::Tools::Output{"Captured the view.", {kPng, kJpeg, kOpaque}};
        }
        if (name == "dump") {
            return Agent::Tools::Output{oversized};
        }
        return Agent::Tools::Output{R"({"error":"no such tool"})", {}, true};
    };

    Agent::Session session;
    REQUIRE(session.submit("Check the FM graph.", config));
    REQUIRE(Wait(session));
    REQUIRE(session.submit("And now?", config));
    REQUIRE(Wait(session));

    const auto files = sandbox.files();
    REQUIRE(files.size() == 1);
    CHECK(std::regex_match(Name(files.front()),
                           std::regex(R"(\d{8}-\d{6}-Nemotron_Local-nvidia_nemotron-3-nano_30b\.md)")));

    const auto text = Read(files.front());
    INFO(text);

    SECTION("the header names the provider, model and effort") {
        CHECK(text.starts_with("# CyberEther agent transcript\n\n- Started: "));
        CHECK(Contains(text, "- Provider: Nemotron Local\n"));
        CHECK(Contains(text, "- Model: nvidia/nemotron-3-nano:30b\n"));
        CHECK(Contains(text, "- Reasoning effort: high\n"));
        CHECK(Contains(text, "- App version: "));
        CHECK(text.find("- Provider: ") < text.find("\n## "));
    }

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)
    SECTION("the file is readable by its owner only") {
        using std::filesystem::perms;
        CHECK(std::filesystem::status(files.front()).permissions() == (perms::owner_read | perms::owner_write));
    }
#endif

    SECTION("entries are timestamped and chronological, and no content breaks a fence") {
        const auto outline = Parse(text);
        CHECK(outline.closed);
        CHECK(Titles(outline) == std::vector<std::string>{
            "User",
            "Assistant reasoning",
            "Assistant",
            "Model round 1",
            "Tool call: inspect",
            "Tool result: inspect",
            "Tool call: capture",
            "Tool result: capture",
            "Tool call: dump",
            "Tool result: dump",
            "Tool call: missing",
            "Tool result: missing",
            "Assistant reasoning",
            "Assistant",
            "Model round 2",
            "User",
            "Assistant",
            "Model round 3",
        });
        CHECK(std::is_sorted(outline.headings.begin(), outline.headings.end(), [](const auto& a, const auto& b) {
            return a.substr(0, 24) < b.substr(0, 24);
        }));
    }

    SECTION("reasoning and replies are written in full") {
        CHECK(Contains(text, "`````text\n" + thought + "`````\n"));
        CHECK(Contains(text, "````text\n" + longThought + "\n````\n"));
        CHECK(Contains(text, "````text\nChecking the graph ```now```.\n````\n"));
        CHECK(Contains(text, "```text\nThe FM graph is healthy.\n```\n"));
        CHECK(Contains(text, "```text\nCheck the FM graph.\n```\n"));
        CHECK(Contains(text, "```text\nAnd now?\n```\n"));
    }

    SECTION("tool calls keep the arguments exactly as sent") {
        CHECK(Contains(text, "- Call id: call-1\n\n````json\n" + inspectArgs + "\n````\n"));
        CHECK(Contains(text, "- Call id: call-4\n\n```json\n```\n"));
    }

    SECTION("tool results carry their status, image sizes and a truncation marker") {
        CHECK(Contains(text, "- Call id: call-1\n- Error: no\n\n`````text\nstate:\n````\nrunning\n````\n`````\n"));
        CHECK(Contains(text, "- Error: no\n- Images: [image 640x480], [image 320x240], [image]\n"));
        CHECK(Contains(text, "```text\n" + std::string(16384, 'x') + "\n```\n\n[truncated, 20015 bytes total]\n"));
        CHECK_FALSE(Contains(text, std::string(16385, 'x')));
        CHECK_FALSE(Contains(text, "<end of result>"));
        CHECK(Contains(text, "- Call id: call-4\n- Error: yes\n\n```text\n{\"error\":\"no such tool\"}\n```\n"));
    }

    SECTION("each model round reports its token usage") {
        CHECK(Contains(text, "- Tokens: prompt 1200, completion 80, cached 1000, reasoning 40\n"
                             "- Tool calls requested: 4\n"));
        CHECK(Contains(text, "- Tokens: prompt 30000, completion 20000, cached 0\n"));
        CHECK(Contains(text, "- Tokens: not reported\n"));
    }

    SECTION("the file grows while the turn is still running") {
        CHECK(Contains(midway, "Check the FM graph."));
        CHECK(Contains(midway, thought));
        CHECK(Contains(midway, "[truncated, 20015 bytes total]"));
        CHECK(Contains(midway, "Tool result: missing"));
        CHECK_FALSE(Contains(midway, "The FM graph is healthy."));
    }

    SECTION("credentials, endpoints, image bytes and the system prompt stay out") {
        CHECK_FALSE(Contains(text, kSecret));
        CHECK_FALSE(Contains(text, "hunter2"));
        CHECK_FALSE(Contains(text, "inference.example.test"));
        CHECK_FALSE(Contains(text, "base64"));
        CHECK_FALSE(Contains(text, "iVBORw0KGgo"));
        CHECK_FALSE(Contains(text, "System instructions"));
    }
}

TEST_CASE("Each new chat starts its own transcript file", "[agent][transcript]") {
    Sandbox sandbox;

    std::atomic<int> replies{0};
    auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
        response.text = "Reply " + std::to_string(++replies) + ".";
        callbacks.onToken(response.text);
        return Result::SUCCESS;
    });

    Agent::Session session;
    for (const char* prompt : {"First chat.", "Second chat.", "Third chat."}) {
        REQUIRE(session.submit(prompt, config));
        REQUIRE(Wait(session));
        session.clear();
    }

    config.model = "other model/v2";
    config.reasoningEffort.clear();
    REQUIRE(session.submit("Fourth chat, first message.", config));
    REQUIRE(Wait(session));
    config.model = kModel;
    REQUIRE(session.submit("Fourth chat, second message.", config));
    REQUIRE(Wait(session));

    const auto files = sandbox.files();
    REQUIRE(files.size() == 4);

    // Chats that start within the same second on the same model get a numeric suffix.
    const std::regex pattern(R"(\d{8}-\d{6}-Nemotron_Local-(nvidia_nemotron-3-nano_30b|other_model_v2)(-\d+)?\.md)");
    std::vector<std::string> prompts;
    for (const auto& file : files) {
        CHECK(std::regex_match(Name(file), pattern));

        const auto text = Read(file);
        const auto titles = Titles(Parse(text));
        REQUIRE_FALSE(titles.empty());
        CHECK(titles.front() == "User");
        CHECK(std::count(titles.begin(), titles.end(), "User") == (Contains(text, "Fourth chat") ? 2 : 1));

        for (const char* prompt : {"First chat.", "Second chat.", "Third chat.", "Fourth chat, first message."}) {
            if (Contains(text, prompt)) {
                prompts.push_back(prompt);
            }
        }

        if (Contains(text, "Fourth chat")) {
            CHECK(Contains(Name(file), "other_model_v2"));
            CHECK(Contains(text, "- Model: other model/v2\n- Reasoning effort: default\n"));
            CHECK(titles == std::vector<std::string>{"User", "Assistant", "Model round 1", "Model changed", "User",
                                                     "Assistant", "Model round 2"});
            CHECK(Contains(text, "- Model: nvidia/nemotron-3-nano:30b\n- Reasoning effort: default\n"));
        }
    }

    std::sort(prompts.begin(), prompts.end());
    CHECK(prompts == std::vector<std::string>{"First chat.", "Fourth chat, first message.", "Second chat.",
                                              "Third chat."});
}

TEST_CASE("Transcripts keep what a stopped, retried or failed turn produced", "[agent][transcript]") {
    Sandbox sandbox;

    SECTION("a cancelled turn leaves the reasoning streamed so far") {
        std::atomic<bool> streaming{false};
        auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
            callbacks.onReasoning("Half a thought", false);
            streaming.store(true);
            while (!callbacks.shouldAbort()) {
                std::this_thread::sleep_for(1ms);
            }
            response.error = {Agent::Provider::Error::Kind::Cancelled, 0, "Cancelled."};
            return Result::ERROR;
        });

        Agent::Session session;
        REQUIRE(session.submit("Think for a long time.", config));
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (!streaming.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(1ms);
        }
        REQUIRE(streaming.load());
        session.cancel();
        REQUIRE(Wait(session));

        REQUIRE(sandbox.files().size() == 1);
        const auto text = Read(sandbox.files().front());
        const auto outline = Parse(text);
        CHECK(outline.closed);
        CHECK(Titles(outline) == std::vector<std::string>{"User", "Assistant reasoning", "Model round 1", "Stopped"});
        CHECK(Contains(text, "```text\nHalf a thought\n```\n"));
        CHECK(Contains(text, "- Error: Cancelled.\n"));
        CHECK(Contains(text, "The turn was cancelled before it finished."));
    }

    SECTION("a blank reply is retried and the retry is recorded") {
        int calls = 0;
        auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
            if (calls++ == 0) {
                return Result::SUCCESS;
            }
            callbacks.onToken("Here now.");
            response.text = "Here now.";
            return Result::SUCCESS;
        });

        Agent::Session session;
        REQUIRE(session.submit("Hello?", config));
        REQUIRE(Wait(session));

        REQUIRE(sandbox.files().size() == 1);
        const auto text = Read(sandbox.files().front());
        CHECK(Titles(Parse(text)) == std::vector<std::string>{"User", "Model round 1", "Status", "Assistant",
                                                              "Model round 2"});
        CHECK(Contains(text, "The model returned an empty reply; asking again."));
    }

    SECTION("an image limit recovery and a provider failure are recorded") {
        int calls = 0;
        auto config = Config(sandbox, [&](const auto&, const auto&, auto& response) {
            if (calls++ == 0) {
                response.error = {Agent::Provider::Error::Kind::InvalidRequest, 400, "At most 1 image(s) may be provided."};
                response.error.imageLimit = 1;
                return Result::ERROR;
            }
            response.error = {Agent::Provider::Error::Kind::Server, 503, "Inference server\nis overloaded"};
            response.error.code = "overloaded";
            response.error.retryAfter = "30";
            return Result::ERROR;
        });

        Agent::Session session;
        REQUIRE(session.submit("Look at this.", config));
        REQUIRE(Wait(session));

        REQUIRE(sandbox.files().size() == 1);
        const auto text = Read(sandbox.files().front());
        const auto outline = Parse(text);
        CHECK(outline.closed);
        CHECK(Titles(outline) == std::vector<std::string>{"User", "Model round 1", "Status", "Model round 2",
                                                          "Error"});
        CHECK(Contains(text, "- Error: At most 1 image(s) may be provided. (HTTP 400)\n"));
        CHECK(Contains(text, "This provider accepts at most 1 image(s) per request"));
        CHECK(Contains(text, "- Error: Inference server is overloaded (HTTP 503) [overloaded] retry after 30\n"));
        CHECK(Contains(text, "```text\nInference server\nis overloaded\n```\n"));
    }

    SECTION("a repeated failing call stops the turn and says why") {
        auto config = Config(sandbox, [&](const auto&, const auto&, auto& response) {
            response.toolCalls = {{"call", "inspect", R"({"id":"gone"})"}};
            return Result::SUCCESS;
        });
        config.executeTool = [](const auto&, const auto&, const auto&) {
            return Agent::Tools::Output{R"({"error":"block not found"})", {}, true};
        };

        Agent::Session session;
        REQUIRE(session.submit("Inspect the missing block.", config));
        REQUIRE(Wait(session));

        REQUIRE(sandbox.files().size() == 1);
        const auto text = Read(sandbox.files().front());
        const auto titles = Titles(Parse(text));
        CHECK(std::count(titles.begin(), titles.end(), "Tool result: inspect") == 3);
        CHECK(titles.back() == "Error");
        CHECK(Contains(text, "[Repeated failing call: third identical failure. The turn was stopped.]"));
        CHECK(Contains(text, "The model sent the same failing 'inspect' tool call 3 times"));
    }
}

TEST_CASE("Transcripts never get in the way of a chat", "[agent][transcript]") {
    Sandbox sandbox;

    auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
        callbacks.onReasoning("Thinking.", false);
        callbacks.onToken("Answer.");
        response.text = "Answer.";
        return Result::SUCCESS;
    });

    SECTION("nothing is written when transcripts are off") {
        config.transcript.enabled = false;

        Agent::Session session;
        REQUIRE(session.submit("Quietly.", config));
        REQUIRE(Wait(session));

        CHECK(session.snapshot().messages.back().segments.back().text == "Answer.");
        CHECK_FALSE(std::filesystem::exists(sandbox.folder));
    }

    SECTION("a folder that cannot be created costs one warning and nothing else") {
        const auto blocker = sandbox.root / "blocker";
        std::ofstream(blocker) << "not a folder";
        config.transcript.folder = Platform::PathToUtf8(blocker / "transcripts");

        JST_LOG_SET_DEBUG_LEVEL(1);
        JST_LOG_LAST_WARNING().clear();

        Agent::Session session;
        REQUIRE(session.submit("First.", config));
        REQUIRE(Wait(session));
        const std::string warning = JST_LOG_LAST_WARNING();
        JST_LOG_LAST_WARNING().clear();

        REQUIRE(session.submit("Second.", config));
        REQUIRE(Wait(session));
        const std::string repeated = JST_LOG_LAST_WARNING();
        JST_LOG_SET_DEBUG_LEVEL(0);

        CHECK(Contains(warning, "Chat transcript disabled"));
        CHECK(repeated.empty());

        const auto snapshot = session.snapshot();
        REQUIRE(snapshot.messages.size() == 4);
        CHECK(snapshot.messages[1].segments.back().kind == Kind::Text);
        CHECK(snapshot.messages[3].segments.back().text == "Answer.");
        CHECK(std::filesystem::is_regular_file(blocker));
        CHECK(sandbox.files().empty());
    }

    SECTION("an empty folder setting is treated the same way") {
        config.transcript.folder.clear();

        Agent::Session session;
        REQUIRE(session.submit("Nowhere.", config));
        REQUIRE(Wait(session));

        CHECK(session.snapshot().messages.back().segments.back().text == "Answer.");
        CHECK_FALSE(std::filesystem::exists(sandbox.folder));
    }
}

TEST_CASE("Transcripts never carry the endpoint or its credentials", "[agent][transcript]") {
    Sandbox sandbox;
    const std::string host = "https://alice:hunter2@inference.example.test";

    const auto clean = [](const std::string& text) {
        CHECK_FALSE(Contains(text, "hunter2"));
        CHECK_FALSE(Contains(text, "alice"));
        CHECK_FALSE(Contains(text, "inference.example.test"));
        CHECK_FALSE(Contains(text, kSecret));
    };

    // These drive the real client. Both requests are refused before a connection is made, on
    // the path every HTTP, timeout, connection and cancellation error of a custom endpoint takes.
    SECTION("a failure reported by the real client names the provider, not its base URL") {
        Agent::Provider::Config endpoint{
            .name = "Local Gateway",
            .kind = Agent::Provider::Kind::Custom,
            .host = host,
            .path = "/v1/chat/completions",
        };
        Agent::Session::Config config;
        config.model = "some-model";
        config.transcript = {true, Platform::PathToUtf8(sandbox.folder)};

        std::string reason;
        SECTION("with a key stored") {
            endpoint.apiKey = [] {
                return std::string(kSecret);
            };
            config.extraBody = "[1]";
            reason = "The model's extra request fields must be a JSON object.";
        }
        SECTION("with no key stored") {
            endpoint.requireApiKey = true;
            reason = "No API key is stored for this provider.";
        }
        config.provider = std::make_shared<Agent::Provider>(endpoint);

        Agent::Session session;
        REQUIRE(session.submit("hello", config));
        REQUIRE(Wait(session));

        REQUIRE(sandbox.files().size() == 1);
        const auto text = Read(sandbox.files().front());
        INFO(text);
        clean(text);
        CHECK(Titles(Parse(text)) == std::vector<std::string>{"User", "Model round 1", "Error"});
        CHECK(Contains(text, "- Error: Local Gateway: " + reason + "\n"));
        CHECK(Contains(text, "```text\nLocal Gateway: " + reason + "\n```\n"));

        // The chat and the log name the endpoint, without the password.
        const auto shown = session.snapshot().messages.back().segments.back().text;
        CHECK(shown == "https://inference.example.test: " + reason);
    }

    SECTION("an error or status that quotes a credentialed URL is scrubbed by the writer") {
        int calls = 0;
        auto config = Config(sandbox, [&](const auto&, const auto&, auto& response) {
            if (calls++ == 0) {
                response.error = {Agent::Provider::Error::Kind::InvalidRequest, 400,
                                  "https://user:hunter2@inference.example.test: At most 1 image(s) may be provided."};
                response.error.imageLimit = 1;
                return Result::ERROR;
            }
            response.error = {Agent::Provider::Error::Kind::Server, 502,
                              "https://user:hunter2@inference.example.test: upstream "
                              "http://alice:p%40ss@proxy.example.test:8080/v1?x=1 refused the request"};
            response.error.code = "https://inference.example.test/errors/bad_gateway";
            response.error.retryAfter = "30";
            return Result::ERROR;
        });

        const auto text = Chat(sandbox, config);
        INFO(text);
        clean(text);
        CHECK_FALSE(Contains(text, "p%40ss"));
        CHECK(Contains(text, "- Error: Nemotron Local: At most 1 image(s) may be provided. (HTTP 400)\n"));
        CHECK(Contains(text, "- Error: Nemotron Local: upstream http://proxy.example.test:8080/v1?x=1 refused the "
                             "request (HTTP 502) [Nemotron Local/errors/bad_gateway] retry after 30\n"));
        CHECK(Contains(text, "```text\nNemotron Local: upstream http://proxy.example.test:8080/v1?x=1 refused the "
                             "request\n```\n"));
    }

    SECTION("what the user, the model and the tools wrote is left as it is") {
        const std::string link = "https://bob:swordfish@files.example.test/iq.bin";
        auto config = Config(sandbox, [&, step = 0](const auto&, const auto& callbacks, auto& response) mutable {
            if (step++ == 0) {
                response.toolCalls = {{"call-1", "inspect", R"({"url":")" + link + R"("})"}};
                return Result::SUCCESS;
            }
            callbacks.onToken("Fetched " + link);
            response.text = "Fetched " + link;
            return Result::SUCCESS;
        });
        config.executeTool = [&](const auto&, const auto&, const auto&) {
            return Agent::Tools::Output{"source: " + link};
        };

        const auto text = Chat(sandbox, config, "Read " + link);
        CHECK(Count(text, link) == 4);
    }
}

TEST_CASE("Tool calls that never ran are still written with their arguments", "[agent][transcript]") {
    Sandbox sandbox;
    const std::string plan = R"({"secret_plan": "replace the  FM demod", "steps": [1, 2]})";

    SECTION("a turn stopped between two calls of one reply") {
        Agent::Session session;
        auto config = Config(sandbox, [&](const auto&, const auto&, auto& response) {
            response.toolCalls = {
                {"call-1", "inspect", R"({"block":"fm"})"},
                {"call-2", "patch", plan},
                {"call-3", "dump", ""},
            };
            return Result::SUCCESS;
        });
        config.executeTool = [&](const auto&, const auto&, const auto&) {
            session.cancel();
            return Agent::Tools::Output{"inspected"};
        };

        REQUIRE(session.submit("Rework the graph.", config));
        REQUIRE(Wait(session));

        REQUIRE(sandbox.files().size() == 1);
        const auto text = Read(sandbox.files().front());
        INFO(text);
        const auto outline = Parse(text);
        CHECK(outline.closed);
        CHECK(Titles(outline) == std::vector<std::string>{
            "User",
            "Model round 1",
            "Tool call: inspect",
            "Tool result: inspect",
            "Tool call (not run): patch",
            "Tool call (not run): dump",
            "Stopped",
        });
        CHECK(Contains(text, "- Tool calls requested: 3\n"));
        CHECK(Contains(text, "- Call id: call-2\n\n```json\n" + plan + "\n```\n"));
        CHECK(Contains(text, "- Call id: call-3\n\n```json\n```\n"));
    }

    SECTION("a turn stopped as the reply with the calls arrives") {
        Agent::Session session;
        auto config = Config(sandbox, [&](const auto&, const auto&, auto& response) {
            response.toolCalls = {{"call-1", "patch", plan}};
            session.cancel();
            return Result::SUCCESS;
        });
        config.executeTool = [](const auto&, const auto&, const auto&) {
            return Agent::Tools::Output{"must not run"};
        };

        REQUIRE(session.submit("Rework the graph.", config));
        REQUIRE(Wait(session));

        REQUIRE(sandbox.files().size() == 1);
        const auto text = Read(sandbox.files().front());
        INFO(text);
        CHECK(Titles(Parse(text)) == std::vector<std::string>{"User", "Model round 1", "Tool call (not run): patch",
                                                              "Stopped"});
        CHECK(Contains(text, "- Call id: call-1\n\n```json\n" + plan + "\n```\n"));
        CHECK_FALSE(Contains(text, "must not run"));
    }

    SECTION("a call sent after the tool-step budget ran out") {
        int rounds = 0;
        auto config = Config(sandbox, [&](const auto& request, const auto&, auto& response) {
            ++rounds;
            response.toolCalls = {{"call-" + std::to_string(rounds), request.allowTools ? "inspect" : "patch",
                                   request.allowTools ? "{}" : plan}};
            return Result::SUCCESS;
        });
        config.maxToolTurns = 2;
        config.executeTool = [](const auto&, const auto&, const auto&) {
            return Agent::Tools::Output{"inspected"};
        };

        const auto text = Chat(sandbox, config, "Keep going.");
        INFO(text);
        const auto outline = Parse(text);
        CHECK(outline.closed);
        CHECK(Titles(outline) == std::vector<std::string>{
            "User",
            "Model round 1",
            "Tool call: inspect",
            "Tool result: inspect",
            "Model round 2",
            "Tool call: inspect",
            "Tool result: inspect",
            "Model round 3",
            "Tool call (not run): patch",
            "Error",
        });
        CHECK(Contains(text, "- Call id: call-3\n\n```json\n" + plan + "\n```\n"));
        CHECK(Contains(text, "The provider returned tool calls while execution was disabled."));
    }
}

TEST_CASE("A transcript file is never shared between conversations", "[agent][transcript]") {
    Sandbox sandbox;

    auto config = Config(sandbox, [](const auto&, const auto& callbacks, auto& response) {
        callbacks.onToken("Answer.");
        response.text = "Answer.";
        return Result::SUCCESS;
    });

    SECTION("chats that start together each get a file of their own") {
        constexpr U64 kChats = 12;

        {
            std::vector<Agent::Session> sessions(kChats);
            for (U64 index = 0; index < kChats; ++index) {
                REQUIRE(sessions[index].submit("Chat " + std::to_string(index) + ".", config));
            }
            for (auto& session : sessions) {
                REQUIRE(Wait(session));
            }
        }

        const auto files = sandbox.files();
        REQUIRE(files.size() == kChats);
        for (const auto& file : files) {
            const auto text = Read(file);
            INFO(Name(file));
            INFO(text);
            CHECK(Count(text, "# CyberEther agent transcript") == 1);
            CHECK(Titles(Parse(text)) == std::vector<std::string>{"User", "Assistant", "Model round 1"});
        }
    }

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)
    SECTION("a name that is already taken is left alone, even when it is a dangling link") {
        // A dangling link does not "exist" to a lookup, so only an exclusive create refuses it.
        const auto elsewhere = sandbox.root / "elsewhere.md";
        std::filesystem::create_directories(sandbox.folder);

        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::vector<std::filesystem::path> others;
        for (std::time_t second = now - 1; second <= now + 5; ++second) {
            std::tm utc{};
            gmtime_r(&second, &utc);
            char stamp[32] = {};
            REQUIRE(std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &utc) != 0);

            const auto stem = std::string(stamp) + "-Nemotron_Local-nvidia_nemotron-3-nano_30b";
            std::filesystem::create_symlink(elsewhere, sandbox.folder / (stem + ".md"));
            others.push_back(sandbox.folder / (stem + "-2.md"));
            std::ofstream(others.back()) << "someone else's chat";
        }

        {
            Agent::Session session;
            REQUIRE(session.submit("Mine.", config));
            REQUIRE(Wait(session));
        }

        CHECK_FALSE(std::filesystem::exists(elsewhere));
        for (const auto& other : others) {
            CHECK(Read(other) == "someone else's chat");
        }

        const auto files = sandbox.files();
        REQUIRE(files.size() == 2 * others.size() + 1);
        const auto mine = std::find_if(files.begin(), files.end(), [](const auto& file) {
            return Contains(Read(file), "Mine.");
        });
        REQUIRE(mine != files.end());
        CHECK(std::regex_match(Name(*mine),
                               std::regex(R"(\d{8}-\d{6}-Nemotron_Local-nvidia_nemotron-3-nano_30b-3\.md)")));
        CHECK(std::filesystem::is_regular_file(std::filesystem::symlink_status(*mine)));
        CHECK(Count(Read(*mine), "# CyberEther agent transcript") == 1);
    }
#endif
}

TEST_CASE("A tool result is capped at 16 KiB whatever its bytes are", "[agent][transcript]") {
    Sandbox sandbox;

    std::string content;
    std::string shown;
    SECTION("bytes that are not text are cut at the limit, not walked back to nothing") {
        content = std::string(20000, '\x80');
        shown = std::string(16384, '\x80');
    }
    SECTION("a character that straddles the limit is left out whole") {
        content = std::string(16382, 'a') + "\xf0\x9f\x93\xa1" + std::string(4000, 'b');
        shown = std::string(16382, 'a');
    }
    SECTION("a character that ends at the limit is kept") {
        content = std::string(16380, 'a') + "\xf0\x9f\x93\xa1" + std::string(4000, 'b');
        shown = std::string(16380, 'a') + "\xf0\x9f\x93\xa1";
    }

    auto config = Config(sandbox, [step = 0](const auto&, const auto& callbacks, auto& response) mutable {
        if (step++ == 0) {
            response.toolCalls = {{"call-1", "dump", "{}"}};
            return Result::SUCCESS;
        }
        callbacks.onToken("Done.");
        response.text = "Done.";
        return Result::SUCCESS;
    });
    config.executeTool = [&](const auto&, const auto&, const auto&) {
        return Agent::Tools::Output{content};
    };

    const auto text = Chat(sandbox, config);
    CHECK(Parse(text).closed);
    CHECK(Contains(text, "```text\n" + shown + "\n```\n\n[truncated, " + std::to_string(content.size()) +
                         " bytes total]\n"));
}

TEST_CASE("A long segment reaches the disk while it is still streaming", "[agent][transcript]") {
    Sandbox sandbox;

    SECTION("reasoning is on disk before its round ends, and reads as one piece afterwards") {
        std::string opening;
        for (int line = 0; opening.size() < 6000; ++line) {
            opening += "Step " + std::to_string(line) + ": weigh the ```fenced``` options again.\n";
        }
        const std::string rest = "A line that is still being written";
        const std::string reply = std::string(5000, 'a');

        std::string early;
        std::string later;
        auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
            callbacks.onReasoning(opening, false);
            early = Read(sandbox.files().front());
            callbacks.onReasoning(rest, false);

            callbacks.onToken(reply);
            later = Read(sandbox.files().front());
            callbacks.onToken("b, and the same line goes on.");
            response.text = reply + "b, and the same line goes on.";
            return Result::SUCCESS;
        });

        const auto text = Chat(sandbox, config);

        // Had the app died here, the reasoning streamed so far would already be in the file.
        CHECK(early.ends_with("Assistant reasoning\n\n````text\n" + opening));
        CHECK_FALSE(Parse(early).closed);
        CHECK(later.ends_with("Assistant\n\n````text\n" + reply));

        const auto outline = Parse(text);
        CHECK(outline.closed);
        CHECK(Titles(outline) == std::vector<std::string>{"User", "Assistant reasoning", "Assistant", "Model round 1"});
        CHECK(Contains(text, "````text\n" + opening + rest + "\n````\n"));
        CHECK(Contains(text, "````text\n" + reply + "b, and the same line goes on.\n````\n"));
    }

    SECTION("a slow stream is saved every couple of seconds") {
        std::string early;
        auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
            callbacks.onReasoning("A slow thought, ", false);
            std::this_thread::sleep_for(2100ms);
            callbacks.onReasoning("one token at a time", false);
            early = Read(sandbox.files().front());
            callbacks.onReasoning(".", false);
            callbacks.onToken("Done.");
            response.text = "Done.";
            return Result::SUCCESS;
        });

        const auto text = Chat(sandbox, config);
        CHECK(early.ends_with("Assistant reasoning\n\n````text\nA slow thought, one token at a time"));
        CHECK(Parse(text).closed);
        CHECK(Contains(text, "````text\nA slow thought, one token at a time.\n````\n"));
    }

    SECTION("backticks that arrive after a checkpoint cannot close the open fence") {
        const std::string opening = std::string(5000, 'x') + "\n";
        std::string early;
        auto config = Config(sandbox, [&](const auto&, const auto& callbacks, auto& response) {
            // The run of four is split over two chunks, and the first half arrives with the checkpoint.
            callbacks.onReasoning(opening + "``", false);
            early = Read(sandbox.files().front());
            callbacks.onReasoning("``\nstill reasoning\n", false);
            callbacks.onToken("Done.");
            response.text = "Done.";
            return Result::SUCCESS;
        });

        const auto text = Chat(sandbox, config);
        INFO(text.substr(text.size() > 600 ? text.size() - 600 : 0));
        CHECK(early.ends_with("````text\n" + opening));

        const auto outline = Parse(text);
        CHECK(outline.closed);
        CHECK(Titles(outline) == std::vector<std::string>{"User", "Assistant reasoning",
                                                          "Assistant reasoning (continued)", "Assistant",
                                                          "Model round 1"});
        CHECK(Contains(text, "````text\n" + opening + "````\n"));
        CHECK(Contains(text, "`````text\n````\nstill reasoning\n`````\n"));
    }
}
