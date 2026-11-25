#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>

#include "../../../src/instance_remote_impl.hh"
#include "jetstream/platform.hh"
#include "jetstream/viewport/adapters/generic.hh"

#include "../render/harness.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/backend/base.hh"
#endif

#include <rtc/rtc.hpp>

#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <openssl/x509.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>

#if defined(JST_OS_LINUX)
#include "agent_provider.hh"
#include "compositor/default/support/agent/providers.hh"
#include "tools/random.hh"
#endif

using namespace Jetstream;
using namespace std::chrono_literals;

namespace {

void SkipRemoteEncodingOnCI() {
    if (std::getenv("CI")) {
        SKIP("Remote encoding tests are disabled in CI.");
    }
}

template<typename Predicate>
bool WaitFor(Predicate predicate, std::chrono::milliseconds timeout = 3s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
    return predicate();
}

struct Stream {
    Instance::Remote::Impl remote;
    std::vector<unsigned char> frame = std::vector<unsigned char>(320 * 240 * 4, 0x7f);
    std::atomic<bool> running = false;
    std::thread producer;

    Stream() {
        SkipRemoteEncodingOnCI();
    }

    void configure() {
        remote.config.broker = {.address = "127.0.0.1", .port = 0};
        remote.size = {320, 240};
        remote.config.video.codec = Instance::Remote::CodecType::H264;
        remote.inputMemoryDevice_ = DeviceType::CPU;
        remote.encoderType = Instance::Remote::EncoderType::Software;
    }

    void start(bool produceFrames = true) {
        configure();
        REQUIRE(remote.createBroker() == Result::SUCCESS);
        remote.started_ = true;
        if (produceFrames) startFrames();
    }

    void startFrames() {
        running = true;
        producer = std::thread([this] {
            while (running) {
                remote.pushNewFrame(frame.data());
                std::this_thread::sleep_for(2ms);
            }
        });
    }

    void stop() {
        running = false;
        remote.destroy();
        if (producer.joinable()) producer.join();
    }

    ~Stream() { stop(); }
};

std::string BrokerTokenHash(const std::string& token) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(token.data()), token.size(), digest);
    std::string encoded(4 * ((sizeof(digest) + 2) / 3), '\0');
    encoded.resize(EVP_EncodeBlock(reinterpret_cast<unsigned char*>(encoded.data()), digest, sizeof(digest)));
    std::replace(encoded.begin(), encoded.end(), '+', '-');
    std::replace(encoded.begin(), encoded.end(), '/', '_');
    while (!encoded.empty() && encoded.back() == '=') {
        encoded.pop_back();
    }
    return encoded;
}

struct BrokerPeer {
    std::unique_ptr<httplib::ws::WebSocketClient> client;

    BrokerPeer(const std::string& origin, const std::string& query = "", const httplib::Headers& headers = {}) {
        std::string url = "ws://" + origin.substr(7) + "/api/v1/remote/signaller" + query;
        client = std::make_unique<httplib::ws::WebSocketClient>(url, headers);
        client->set_read_timeout(3s);
    }

    httplib::ws::Result connect() {
        return client->connect();
    }

    void send(const nlohmann::json& message) {
        REQUIRE(client->send(message.dump()));
    }

    nlohmann::json next(const std::string& type) {
        std::string payload;
        while (client->read(payload) == httplib::ws::Text) {
            auto message = nlohmann::json::parse(payload);
            if (message.value("type", "") == type) {
                return message;
            }
        }
        FAIL("Broker did not send '" << type << "'.");
        return {};
    }

    bool closed() {
        std::string payload;
        for (int attempt = 0; attempt < 8; ++attempt) {
            const auto result = client->read(payload);
            if (result != httplib::ws::Text && result != httplib::ws::Timeout) {
                return true;
            }
        }
        return false;
    }
};

void WriteSelfSignedCertificate(const std::filesystem::path& certificate, const std::filesystem::path& key) {
    EVP_PKEY* pkey = EVP_EC_gen("P-256");
    REQUIRE(pkey);
    X509* x509 = X509_new();
    REQUIRE(x509);
    ASN1_INTEGER_set(X509_get_serialNumber(x509), 1);
    X509_gmtime_adj(X509_getm_notBefore(x509), 0);
    X509_gmtime_adj(X509_getm_notAfter(x509), 3600);
    X509_set_pubkey(x509, pkey);
    X509_NAME* name = X509_get_subject_name(x509);
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, reinterpret_cast<const unsigned char*>("localhost"), -1, -1, 0);
    X509_set_issuer_name(x509, name);
    REQUIRE(X509_sign(x509, pkey, EVP_sha256()) > 0);

    FILE* file = std::fopen(certificate.string().c_str(), "wb");
    REQUIRE(file);
    CHECK(PEM_write_X509(file, x509) == 1);
    std::fclose(file);
    file = std::fopen(key.string().c_str(), "wb");
    REQUIRE(file);
    CHECK(PEM_write_PrivateKey(file, pkey, nullptr, nullptr, 0, nullptr, nullptr) == 1);
    std::fclose(file);

    X509_free(x509);
    EVP_PKEY_free(pkey);
}

struct InputView {
    const Render::InputState& input;
    std::vector<Render::InputState::Event> collected;

    bool keyDown(KeyCode key) const { return input.keyDown(key); }
    bool windowFocused() const { return input.windowFocused(); }
    const KeyModifiers& modifiers() const { return input.modifiers(); }
    const std::array<bool, 3>& buttons() const { return input.buttons(); }
    const std::optional<Extent2D<F32>>& pointer() const { return input.pointer(); }
    const std::vector<Render::InputState::Event>& events() const { return collected; }
};

struct InputContext {
    RenderTest::Window window;

    explicit InputContext(bool macOS = false) {
        window.viewport.macOS = macOS;
        REQUIRE(window.open() == Result::SUCCESS);
    }

    void attach(Instance::Remote::Impl& remote) {
        remote.viewport = &window.viewport;
    }

    template<typename Check>
    void frame(Instance::Remote::Impl& remote, Check check) {
        REQUIRE(remote.processInput() == Result::SUCCESS);
        InputView view{window.input(), {}};
        do {
            REQUIRE(window.inject({}) == Result::SUCCESS);
            const auto& events = window.input().events();
            view.collected.insert(view.collected.end(), events.begin(), events.end());
        } while (window.input().hasPending());
        check(view);
    }
};

U64 TextCount(const InputView& state) {
    return RenderTest::Only<TextEvent>(state.events()).size();
}

char32_t FirstText(const InputView& state) {
    const auto text = RenderTest::Only<TextEvent>(state.events());
    return text.empty() ? 0 : text.front().codepoint;
}

F32 WheelY(const InputView& state) {
    F32 total = 0.0f;
    for (const auto& mouse : RenderTest::Only<MouseEvent>(state.events())) {
        if (mouse.type == MouseEventType::Scroll) {
            total += mouse.scroll.y;
        }
    }
    return total;
}

F32 PointerX(const InputView& state) {
    return state.pointer().has_value() ? state.pointer()->x : -1.0f;
}

void SendInput(Instance::Remote::Impl& remote, const std::string& sessionId, const nlohmann::json& input) {
    remote.handleControlMessage(sessionId, input.dump());
}

struct Viewer {
    Instance::Remote::Impl& remote;
    BrokerPeer signaller;
    std::string sessionId;
    std::shared_ptr<rtc::PeerConnection> connection;
    std::shared_ptr<rtc::Track> track;
    std::shared_ptr<rtc::RtcpReceivingSession> rtcp;
    std::shared_ptr<rtc::DataChannel> control;
    std::function<std::string(const std::string&)> editAnswer;
    bool holdAnswer = false;
    nlohmann::json heldAnswer;
    std::atomic<unsigned> packets = 0;
    std::atomic<bool> ended = false;
    std::atomic<bool> running = false;
    std::thread reader;
    std::mutex mutex;
    std::string offer;
    std::vector<std::string> candidates;

    explicit Viewer(Stream& stream)
        : remote(stream.remote),
          signaller(stream.remote.broker->origin(), "?tokenHash=" + BrokerTokenHash(stream.remote.consumerToken)) {}

    void join() {
        REQUIRE(signaller.connect());
        signaller.send({{"type", "setPeerStatus"}, {"roles", {"listener"}}, {"meta", {{"token", remote.consumerToken}}}});
        sessionId = signaller.next("sessionRegistered")["sessionId"];
        signaller.client->set_read_timeout(100ms);
    }

    void prepare() {
        join();
        connection = std::make_shared<rtc::PeerConnection>(rtc::Configuration{});
        connection->onLocalDescription([this](rtc::Description description) {
            std::string sdp(description);
            if (editAnswer) {
                sdp = editAnswer(sdp);
            }
            const nlohmann::json message = {{"type", "peer"}, {"sessionId", sessionId},
                                            {"sdp", {{"type", description.typeString()}, {"sdp", sdp}}}};
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (holdAnswer) {
                    heldAnswer = message;
                    return;
                }
            }
            send(message);
        });
        connection->onLocalCandidate([this](rtc::Candidate candidate) {
            send({{"type", "peer"}, {"sessionId", sessionId},
                  {"ice", {{"candidate", candidate.candidate()}, {"sdpMid", candidate.mid()}}}});
        });
        connection->onTrack([this](std::shared_ptr<rtc::Track> incoming) {
            auto session = std::make_shared<rtc::RtcpReceivingSession>();
            incoming->setMediaHandler(session);
            incoming->onMessage([this](rtc::binary) {
                ++packets;
            }, nullptr);
            std::lock_guard<std::mutex> lock(mutex);
            track = std::move(incoming);
            rtcp = std::move(session);
        });
        connection->onDataChannel([this](std::shared_ptr<rtc::DataChannel> channel) {
            std::lock_guard<std::mutex> lock(mutex);
            control = std::move(channel);
        });

        running = true;
        reader = std::thread([this] { read(); });
    }

    void connect() {
        prepare();
        REQUIRE(WaitFor([&] {
            std::lock_guard<std::mutex> lock(remote.remoteStateMutex);
            return std::ranges::find(remote.waitlist_, sessionId) != remote.waitlist_.end();
        }));
        REQUIRE(remote.approveClient(sessionId) == Result::SUCCESS);
    }

    void send(const nlohmann::json& message) {
        std::lock_guard<std::mutex> lock(mutex);
        (void)signaller.client->send(message.dump());
    }

    void read() {
        std::string payload;
        while (running) {
            const auto result = signaller.client->read(payload);
            if (result == httplib::ws::Timeout) {
                continue;
            }
            if (result != httplib::ws::Text) {
                break;
            }

            const auto message = nlohmann::json::parse(payload, nullptr, false);
            const std::string type = message.value("type", "");
            if (type == "endSession") {
                ended = true;
                continue;
            }
            if (type != "peer") {
                continue;
            }

            try {
                if (message.contains("sdp")) {
                    const std::string sdp = message["sdp"]["sdp"];
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        offer = sdp;
                    }
                    connection->setRemoteDescription(rtc::Description(sdp, message["sdp"]["type"].get<std::string>()));
                }
                if (message.contains("ice")) {
                    const std::string candidate = message["ice"]["candidate"];
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        candidates.push_back(candidate);
                    }
                    connection->addRemoteCandidate(rtc::Candidate(candidate, message["ice"].value("sdpMid", "")));
                }
            } catch (const std::exception&) {
            }
        }
    }

    bool answered() {
        std::lock_guard<std::mutex> lock(mutex);
        return !heldAnswer.is_null();
    }

    void releaseAnswer() {
        nlohmann::json message;
        {
            std::lock_guard<std::mutex> lock(mutex);
            holdAnswer = false;
            message = std::move(heldAnswer);
        }
        send(message);
    }

    bool connected() const {
        return connection && connection->state() == rtc::PeerConnection::State::Connected;
    }

    std::string description() {
        std::lock_guard<std::mutex> lock(mutex);
        return offer;
    }

    bool requestKeyframe() {
        std::lock_guard<std::mutex> lock(mutex);
        return track && track->requestKeyframe();
    }

    bool senderReport() {
        std::lock_guard<std::mutex> lock(mutex);
        return rtcp && rtcp->getSyncTimestamps().ntpTimestamp != 0;
    }

    void close() {
        running = false;
        if (reader.joinable()) reader.join();
        if (connection) {
            connection->resetCallbacks();
        }
        std::shared_ptr<rtc::Track> incoming;
        {
            std::lock_guard<std::mutex> lock(mutex);
            incoming = track;
        }
        if (incoming) {
            incoming->resetCallbacks();
        }
        if (connection) {
            connection->close();
        }
    }

    ~Viewer() {
        close();
        if (!sessionId.empty()) {
            send({{"type", "endSession"}, {"sessionId", sessionId}});
        }
        remote.destroyWebRtcSession(sessionId);
    }
};

}  // namespace

TEST_CASE("Only the first remote consumer can control the interface",
          "[core][application][remote][input]") {
    const std::string sender = GENERATE("z-first", "a-second", "unknown");
    const bool controls = sender == "z-first";
    const bool macOS = GENERATE(false, true);
    InputContext input(macOS);
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    input.attach(remote);
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);
    REQUIRE(remote.createWebRtcSession("a-second", "second") == Result::SUCCESS);

    SendInput(remote, sender, {{"kind", "keyboard"}, {"action", "down"}, {"code", "KeyA"}, {"ctrlKey", true}});
    SendInput(remote, sender, {{"kind", "mouse"}, {"act", "down"}, {"button", 0}, {"x", 0.5}, {"y", 0.5}});
    SendInput(remote, sender, {{"kind", "wheel"}, {"deltaY", -100}});
    SendInput(remote, sender, {{"kind", "text"}, {"text", "x"}});
    input.frame(remote, [&](const InputView& state) {
        CHECK(state.keyDown(KeyCode::A) == controls);
        CHECK(state.modifiers().control == controls);
        CHECK_FALSE(state.modifiers().super);
        CHECK(state.buttons()[0] == (controls && !macOS));
        CHECK(state.buttons()[1] == (controls && macOS));
        CHECK(WheelY(state) == (controls ? 1.0f : 0.0f));
        CHECK(TextCount(state) == (controls ? 1u : 0u));
        CHECK((PointerX(state) == 160.0f) == controls);
    });

    remote.enqueueInputReset(sender);
    input.frame(remote, [](const InputView& state) {
        CHECK_FALSE(state.keyDown(KeyCode::A));
        CHECK_FALSE(state.modifiers().control);
        CHECK_FALSE(state.modifiers().super);
        CHECK_FALSE(state.buttons()[0]);
        CHECK_FALSE(state.buttons()[1]);
    });
}

TEST_CASE("Remote pointer coordinates follow the viewport framebuffer scale",
          "[core][application][remote][input]") {
    InputContext input;
    input.window.viewport.display = {800.0f, 600.0f};
    input.window.viewport.scale = 2.0f;
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    input.attach(remote);
    remote.size = {1600, 1200};
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);

    SendInput(remote, "z-first", {{"kind", "mouse"}, {"act", "move"}, {"x", 0.5}, {"y", 0.5}});
    input.frame(remote, [](const InputView& state) {
        REQUIRE(state.pointer().has_value());
        CHECK(state.pointer()->x == 400.0f);
        CHECK(state.pointer()->y == 300.0f);
    });
}

TEST_CASE("Remote control keeps working while the host window is unfocused",
          "[core][application][remote][input]") {
    InputContext input;
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    input.attach(remote);
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);

    input.window.viewport.input().push(Render::InputQueue::Focus{false});
    input.window.viewport.input().push(Render::InputQueue::Button{MouseButton::Left, true});
    SendInput(remote, "z-first", {{"kind", "mouse"}, {"act", "down"}, {"button", 0}, {"x", 0.5}, {"y", 0.5}});
    SendInput(remote, "z-first", {{"kind", "keyboard"}, {"action", "down"}, {"code", "KeyA"}});
    input.frame(remote, [](const InputView& state) {
        CHECK(state.windowFocused());
        CHECK(state.buttons()[0]);
        CHECK(state.keyDown(KeyCode::A));
    });
}

TEST_CASE("Remote control follows connection order and releases input on handoff",
          "[core][application][remote][input]") {
    InputContext input;
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    input.attach(remote);
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);
    REQUIRE(remote.createWebRtcSession("m-second", "second") == Result::SUCCESS);
    REQUIRE(remote.createWebRtcSession("a-third", "third") == Result::SUCCESS);
    // Duplicate start messages must not create another position in the queue.
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);

    SendInput(remote, "z-first", {{"kind", "keyboard"}, {"action", "down"}, {"code", "KeyA"}, {"shiftKey", true}});
    SendInput(remote, "z-first", {{"kind", "mouse"}, {"act", "down"}, {"button", 0}, {"x", 0.5}, {"y", 0.5}});
    input.frame(remote, [](const InputView& state) {
        CHECK(state.keyDown(KeyCode::A));
        CHECK(state.modifiers().shift);
        CHECK(state.buttons()[0]);
    });

    // Already queued input from the departed controller and spectator input
    // received before promotion must not be applied after handoff.
    SendInput(remote, "z-first", {{"kind", "text"}, {"text", "stale"}});
    SendInput(remote, "z-first", {{"kind", "mouse"}, {"act", "move"}, {"x", 0.0}, {"y", 0.0}});
    SendInput(remote, "m-second", {{"kind", "text"}, {"text", "early"}});
    remote.destroyWebRtcSession("z-first");
    SendInput(remote, "z-first", {{"kind", "text"}, {"text", "late"}});
    SendInput(remote, "a-third", {{"kind", "text"}, {"text", "viewer"}});
    SendInput(remote, "m-second", {{"kind", "keyboard"}, {"action", "down"}, {"code", "KeyB"}});
    SendInput(remote, "m-second", {{"kind", "text"}, {"text", "b"}});
    input.frame(remote, [](const InputView& state) {
        CHECK_FALSE(state.keyDown(KeyCode::A));
        CHECK(state.keyDown(KeyCode::B));
        CHECK_FALSE(state.modifiers().shift);
        CHECK_FALSE(state.buttons()[0]);
        CHECK(PointerX(state) == 160.0f);
        CHECK(TextCount(state) == 1u);
        CHECK(FirstText(state) == U'b');
    });

    // A reconnect joins at the back, even if it was the original controller.
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);
    remote.destroyWebRtcSession("m-second");
    SendInput(remote, "z-first", {{"kind", "text"}, {"text", "reconnected"}});
    SendInput(remote, "a-third", {{"kind", "text"}, {"text", "c"}});
    input.frame(remote, [](const InputView& state) {
        CHECK_FALSE(state.keyDown(KeyCode::B));
        CHECK(TextCount(state) == 1u);
        CHECK(FirstText(state) == U'c');
    });

    // Removing a spectator cannot release the controller's held input.
    SendInput(remote, "a-third", {{"kind", "mouse"}, {"act", "down"}, {"button", 0}});
    remote.destroyWebRtcSession("z-first");
    input.frame(remote, [](const InputView& state) { CHECK(state.buttons()[0]); });

    stream.stop();
    stream.start();
    REQUIRE(remote.createWebRtcSession("new-first", "new") == Result::SUCCESS);
    SendInput(remote, "new-first", {{"kind", "text"}, {"text", "d"}});
    input.frame(remote, [](const InputView& state) {
        CHECK_FALSE(state.buttons()[0]);
        CHECK(TextCount(state) == 1u);
        CHECK(FirstText(state) == U'd');
    });
}

TEST_CASE("Broker disconnect stops remote streaming and releases its room and input state",
          "[core][application][remote][broker]") {
    InputContext input;
    Stream stream;
    auto& remote = stream.remote;
    input.attach(remote);
    remote.config.broker = {.address = "127.0.0.1", .port = 0};
    remote.size = {320, 240};
    std::string previousRoom;

    for (unsigned i = 1; i <= 2; ++i) {
        CAPTURE(i);
        remote.inputMemoryDevice_ = DeviceType::CPU;
        remote.encoderType = Instance::Remote::EncoderType::Software;
        REQUIRE(remote.createBroker() == Result::SUCCESS);
        remote.started_ = true;
        REQUIRE_FALSE(remote.roomId_.empty());
        REQUIRE(remote.roomId_ != previousRoom);
        previousRoom = remote.roomId_;

        BrokerPeer viewer(remote.broker->origin(), "?tokenHash=" + BrokerTokenHash(remote.consumerToken));
        REQUIRE(viewer.connect());
        viewer.send({{"type", "setPeerStatus"}, {"roles", {"listener"}}, {"meta", {{"token", remote.consumerToken}}}});
        viewer.next("sessionRegistered");
        REQUIRE(WaitFor([&] {
            std::lock_guard<std::mutex> lock(remote.remoteStateMutex);
            return remote.clients_.empty() && remote.waitlist_.size() == 1;
        }));
        REQUIRE(remote.createWebRtcSession("viewer", "loopback") == Result::SUCCESS);
        SendInput(remote, "viewer", {{"kind", "keyboard"}, {"action", "down"},
                                      {"code", "KeyA"}, {"shiftKey", true}});
        SendInput(remote, "viewer", {{"kind", "mouse"}, {"act", "down"}, {"button", 0}});
        input.frame(remote, [](const InputView& state) {
            CHECK(state.keyDown(KeyCode::A));
            CHECK(state.modifiers().shift);
            CHECK(state.buttons()[0]);
        });
        REQUIRE(remote.captureFrame(true) == Result::SUCCESS);
        REQUIRE(remote.started_);
        REQUIRE(remote.broker->destroy() == Result::SUCCESS);
        REQUIRE(WaitFor([&] { return !remote.signallerRunning; }));
        CHECK(viewer.closed());
        REQUIRE(remote.captureFrame(true) == Result::SUCCESS);

        CHECK_FALSE(remote.started_);
        CHECK_FALSE(remote.streaming);
        CHECK_FALSE(remote.encoder);
        CHECK_FALSE(remote.broker);
        CHECK_FALSE(remote.signallerClient);
        CHECK_FALSE(remote.signallerThread.joinable());
        CHECK(remote.sessions.empty());
        CHECK(remote.roomId_.empty());
        CHECK(remote.consumerToken.empty());
        CHECK(remote.inviteUrl_.empty());
        CHECK(remote.clients_.empty());
        CHECK(remote.waitlist_.empty());
        CHECK(remote.inputSessionOrder.empty());
        input.frame(remote, [](const InputView& state) {
            CHECK_FALSE(state.keyDown(KeyCode::A));
            CHECK_FALSE(state.modifiers().shift);
            CHECK_FALSE(state.buttons()[0]);
        });
        REQUIRE(remote.destroy() == Result::SUCCESS);
    }
}

TEST_CASE("Remote capture honors the configured frame rate without catch-up bursts",
          "[core][application][remote][capture]") {
    const U32 renderFps = GENERATE(30u, 60u, 144u);
    const U32 captureFps = GENERATE(30u, 60u);
    CAPTURE(renderFps, captureFps);
    struct Capture final : Viewport::FrameCapture {
        unsigned frames = 0;
        Result create(Viewport::Generic*, const DeviceType&) override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        Result stop() override { return Result::SUCCESS; }
        Result captureFrame() override { ++frames; return Result::SUCCESS; }
        Result getFrameData(Tensor&) override { return Result::ERROR; }
        Result releaseFrame() override { return Result::SUCCESS; }
    };
    Instance::Remote::Impl remote;
    remote.config.video.framerate = captureFps;
    auto capture = std::make_unique<Capture>();
    auto* counter = capture.get();
    remote.frameCapture = std::move(capture);
    const auto start = std::chrono::steady_clock::time_point(1s);
    for (U32 frame = 0; frame < renderFps * 2; ++frame) {
        const auto now = start + std::chrono::nanoseconds(frame * 1'000'000'000ull / renderFps);
        REQUIRE(remote.captureFrame(true, now) == Result::SUCCESS);
    }
    CHECK(counter->frames == std::min(renderFps, captureFps) * 2);

    const auto before = counter->frames;
    const auto resumed = start + 10s;
    REQUIRE(remote.captureFrame(true, resumed) == Result::SUCCESS);
    for (unsigned i = 0; i < 8; ++i) {
        REQUIRE(remote.captureFrame(true, resumed + 1ms) == Result::SUCCESS);
    }
    CHECK(counter->frames == before + 1);
    REQUIRE(remote.captureFrame(true, resumed + std::chrono::nanoseconds(1'000'000'000ull / captureFps)) == Result::SUCCESS);
    CHECK(counter->frames == before + 2);
}

TEST_CASE("Remote capture skips frames that presented nothing", "[core][application][remote][capture]") {
    struct Capture final : Viewport::FrameCapture {
        unsigned frames = 0;
        Result create(Viewport::Generic*, const DeviceType&) override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        Result stop() override { return Result::SUCCESS; }
        Result captureFrame() override { ++frames; return Result::SUCCESS; }
        Result getFrameData(Tensor&) override { return Result::ERROR; }
        Result releaseFrame() override { return Result::SUCCESS; }
    };
    Instance::Remote::Impl remote;
    remote.config.video.framerate = 60;
    auto capture = std::make_unique<Capture>();
    auto* counter = capture.get();
    remote.frameCapture = std::move(capture);
    const auto start = std::chrono::steady_clock::time_point(1s);
    REQUIRE(remote.captureFrame(false, start) == Result::SUCCESS);
    REQUIRE(remote.captureFrame(false, start + 1s) == Result::SUCCESS);
    CHECK(counter->frames == 0);
    REQUIRE(remote.captureFrame(true, start + 2s) == Result::SUCCESS);
    CHECK(counter->frames == 1);
}

TEST_CASE("A redraw inside the capture interval is captured once the interval elapses",
          "[core][application][remote][capture]") {
    struct Capture final : Viewport::FrameCapture {
        unsigned frames = 0;
        Result create(Viewport::Generic*, const DeviceType&) override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        Result stop() override { return Result::SUCCESS; }
        Result captureFrame() override { ++frames; return Result::SUCCESS; }
        Result getFrameData(Tensor&) override { return Result::ERROR; }
        Result releaseFrame() override { return Result::SUCCESS; }
    };
    RenderTest::Viewport viewport;
    Instance::Remote::Impl remote;
    remote.viewport = &viewport;
    remote.config.video.framerate = 60;
    auto capture = std::make_unique<Capture>();
    auto* counter = capture.get();
    remote.frameCapture = std::move(capture);
    (void)viewport.consumeRedraw();

    const auto start = std::chrono::steady_clock::time_point(1s);
    REQUIRE(remote.captureFrame(true, start) == Result::SUCCESS);
    CHECK(counter->frames == 1);
    REQUIRE(remote.captureFrame(true, start + 1ms) == Result::SUCCESS);
    CHECK(counter->frames == 1);
    REQUIRE(remote.captureFrame(false, start + 2ms) == Result::SUCCESS);
    CHECK_FALSE(viewport.consumeRedraw());

    REQUIRE(remote.captureFrame(false, start + 1s) == Result::SUCCESS);
    CHECK(counter->frames == 1);
    CHECK(viewport.consumeRedraw());
    REQUIRE(remote.captureFrame(true, start + 1s + 1ms) == Result::SUCCESS);
    CHECK(counter->frames == 2);

    REQUIRE(remote.captureFrame(false, start + 2s) == Result::SUCCESS);
    CHECK_FALSE(viewport.consumeRedraw());
    CHECK(counter->frames == 2);
}

TEST_CASE("A keyframe request on an idle window asks for a frame and captures it",
          "[core][application][remote][capture]") {
    struct Capture final : Viewport::FrameCapture {
        unsigned frames = 0;
        Result create(Viewport::Generic*, const DeviceType&) override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        Result stop() override { return Result::SUCCESS; }
        Result captureFrame() override { ++frames; return Result::SUCCESS; }
        Result getFrameData(Tensor&) override { return Result::ERROR; }
        Result releaseFrame() override { return Result::SUCCESS; }
    };
    RenderTest::Viewport viewport;
    Instance::Remote::Impl remote;
    remote.viewport = &viewport;
    remote.config.video.framerate = 60;
    auto capture = std::make_unique<Capture>();
    auto* counter = capture.get();
    remote.frameCapture = std::move(capture);
    (void)viewport.consumeRedraw();

    const auto start = std::chrono::steady_clock::time_point(1s);
    remote.requestKeyframe();
    CHECK(viewport.consumeRedraw());
    REQUIRE(remote.captureFrame(false, start) == Result::SUCCESS);
    CHECK(counter->frames == 0);
    CHECK(viewport.consumeRedraw());
    REQUIRE(remote.captureFrame(true, start + 1ms) == Result::SUCCESS);
    CHECK(counter->frames == 1);
    REQUIRE(remote.captureFrame(false, start + 1s) == Result::SUCCESS);
    CHECK_FALSE(viewport.consumeRedraw());
}

TEST_CASE("A busy capture slot keeps the frame pending until a capture is accepted",
          "[core][application][remote][capture]") {
    struct Capture final : Viewport::FrameCapture {
        unsigned attempts = 0;
        unsigned frames = 0;
        Result create(Viewport::Generic*, const DeviceType&) override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        Result stop() override { return Result::SUCCESS; }
        Result captureFrame() override {
            if (++attempts == 1) {
                return Result::SKIP;
            }
            ++frames;
            return Result::SUCCESS;
        }
        Result getFrameData(Tensor&) override { return Result::ERROR; }
        Result releaseFrame() override { return Result::SUCCESS; }
    };
    RenderTest::Viewport viewport;
    Instance::Remote::Impl remote;
    remote.viewport = &viewport;
    remote.config.video.framerate = 60;
    auto capture = std::make_unique<Capture>();
    auto* counter = capture.get();
    remote.frameCapture = std::move(capture);
    (void)viewport.consumeRedraw();

    const auto start = std::chrono::steady_clock::time_point(1s);
    REQUIRE(remote.captureFrame(true, start) == Result::SUCCESS);
    CHECK(counter->frames == 0);
    REQUIRE(remote.captureFrame(false, start + 1ms) == Result::SUCCESS);
    CHECK(viewport.consumeRedraw());
    REQUIRE(remote.captureFrame(true, start + 2ms) == Result::SUCCESS);
    CHECK(counter->frames == 1);
    REQUIRE(remote.captureFrame(false, start + 1s) == Result::SUCCESS);
    CHECK_FALSE(viewport.consumeRedraw());
}

TEST_CASE("Viewer keyframe requests wake idle capture across stream restarts",
          "[core][application][remote][stream][capture][keyframe]") {
    RenderTest::Viewport viewport;
    Stream stream;
    auto& remote = stream.remote;
    remote.viewport = &viewport;

    for (unsigned i = 0; i < 2; ++i) {
        CAPTURE(i);
        stream.start(false);
        {
            Viewer viewer(stream);
            viewer.connect();
            REQUIRE(WaitFor([&] {
                REQUIRE(remote.pushNewFrame(stream.frame.data()) == Result::SUCCESS);
                return viewer.packets > 0;
            }, 10s));
            remote.captureRequested = false;
            remote.forceKeyframe = false;
            (void)viewport.consumeRedraw();
            REQUIRE(viewer.requestKeyframe());
            REQUIRE(WaitFor([&] { return remote.captureRequested.load(); }));
            CHECK(remote.forceKeyframe.load());
            bool redraw = false;
            CHECK(WaitFor([&] {
                redraw = redraw || viewport.consumeRedraw();
                return redraw;
            }));
        }
        stream.stop();
    }
}

TEST_CASE("An idle remote viewer requests a fresh frame after connecting",
          "[core][application][remote][stream][capture][keyframe]") {
    struct Capture final : Viewport::FrameCapture {
        unsigned frames = 0;
        Result create(Viewport::Generic*, const DeviceType&) override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        Result stop() override { return Result::SUCCESS; }
        Result captureFrame() override { ++frames; return Result::SUCCESS; }
        Result getFrameData(Tensor&) override { return Result::ERROR; }
        Result releaseFrame() override { return Result::SUCCESS; }
    };
    RenderTest::Viewport viewport;
    Stream stream;
    stream.start(false);
    auto& remote = stream.remote;
    remote.viewport = &viewport;
    auto capture = std::make_unique<Capture>();
    auto* counter = capture.get();
    remote.frameCapture = std::move(capture);

    Viewer viewer(stream);
    viewer.holdAnswer = true;
    viewer.connect();
    REQUIRE(WaitFor([&] { return viewer.answered(); }));
    REQUIRE(remote.captureFrame(true) == Result::SUCCESS);
    CHECK(counter->frames == 1);
    REQUIRE(remote.pushNewFrame(stream.frame.data()) == Result::SUCCESS);
    remote.captureRequested = false;
    (void)viewport.consumeRedraw();

    viewer.releaseAnswer();
    REQUIRE(WaitFor([&] { return viewer.connected(); }, 10s));
    REQUIRE(WaitFor([&] { return remote.captureRequested.load(); }));
    bool redraw = false;
    REQUIRE(WaitFor([&] {
        redraw = redraw || viewport.consumeRedraw();
        return redraw;
    }));
    REQUIRE(remote.captureFrame(true, remote.nextCaptureTime) == Result::SUCCESS);
    CHECK(counter->frames == 2);
    REQUIRE(WaitFor([&] {
        REQUIRE(remote.pushNewFrame(stream.frame.data()) == Result::SUCCESS);
        return viewer.packets > 0;
    }));
}

TEST_CASE("Remote encoder turns BGRA frames into keyframes on request",
          "[core][application][remote][stream][encoder]") {
    SkipRemoteEncodingOnCI();
    const U32 framerate = GENERATE(30u, 60u);
    CAPTURE(framerate);

    Instance::Remote::Encoder encoder;
    REQUIRE(encoder.create({
        .codec = Instance::Remote::CodecType::H264,
        .encoder = Instance::Remote::EncoderType::Software,
        .device = DeviceType::CPU,
        .size = {320, 240},
        .framerate = framerate,
    }) == Result::SUCCESS);

    std::vector<unsigned char> frame(320 * 240 * 4, 0x7f);
    std::vector<Instance::Remote::Encoder::Packet> packets;
    for (U64 index = 0; index < 8; ++index) {
        const auto timestamp = std::chrono::microseconds(index * 1'000'000 / framerate);
        REQUIRE(encoder.encode(frame.data(), timestamp, index == 0 || index == 5, packets) == Result::SUCCESS);
    }

    REQUIRE(packets.size() == 8);
    for (U64 index = 0; index < packets.size(); ++index) {
        CAPTURE(index);
        CHECK(packets[index].timestamp == std::chrono::microseconds(index * 1'000'000 / framerate));
        CHECK(packets[index].keyframe == (index == 0 || index == 5));
        REQUIRE(packets[index].data.size() > 4);
        CHECK(packets[index].data[0] == 0x00);
        CHECK(packets[index].data[1] == 0x00);
    }
}

TEST_CASE("Remote offers advertise the video codec and the control channel",
          "[core][application][remote][stream][sdp]") {
    Stream stream;
    stream.start(false);

    // Disconnecting before negotiation must not leave a stale session behind,
    // including when a later viewer reuses the same identifier.
    stream.remote.handleStartSession({{"sessionId", "viewer"}, {"peerId", "cancelled"}});
    stream.remote.destroyWebRtcSession("viewer");
    CHECK(stream.remote.sessions.empty());

    Viewer viewer(stream);
    viewer.connect();
    REQUIRE(WaitFor([&] { return !viewer.description().empty(); }));

    const std::string offer = viewer.description();
    CHECK(offer.find("m=video") != std::string::npos);
    CHECK(offer.find("m=application") != std::string::npos);
    CHECK(offer.find("H264/90000") != std::string::npos);
    CHECK(offer.find("profile-level-id=42e029") != std::string::npos);
    CHECK(offer.find("packetization-mode=1") != std::string::npos);

    stream.startFrames();
    REQUIRE(WaitFor([&] { return viewer.packets >= 20; }, 10s));
    REQUIRE(WaitFor([&] {
        std::lock_guard<std::mutex> lock(viewer.mutex);
        return viewer.control && viewer.control->label() == "control";
    }));
}

TEST_CASE("Remote SDP validation closes rejected sessions and permits valid negotiations",
          "[core][application][remote][stream][sdp]") {
    const bool valid = GENERATE(false, true);
    CAPTURE(valid);
    Stream stream;
    stream.start();
    Viewer observer(stream);
    observer.connect();
    REQUIRE(WaitFor([&] { return observer.packets >= 20; }, 10s));

    Viewer viewer(stream);
    if (!valid) {
        viewer.editAnswer = [](const std::string& sdp) {
            std::string edited;
            std::istringstream lines(sdp);
            for (std::string line; std::getline(lines, line);) {
                if (!line.starts_with("a=ice-pwd")) {
                    edited += line + "\n";
                }
            }
            return edited;
        };
    }
    viewer.connect();

    if (valid) {
        REQUIRE(WaitFor([&] { return viewer.packets >= 20; }, 10s));
    } else {
        REQUIRE(WaitFor([&] { return viewer.ended.load(); }));
        CHECK_FALSE(stream.remote.findSession(viewer.sessionId));
    }

    CHECK(stream.remote.applyRemoteDescription(observer.sessionId, "rollback", "v=0") == Result::ERROR);
    CHECK(stream.remote.applyRemoteDescription("unknown", "answer", "v=0") == Result::ERROR);
    const unsigned before = observer.packets;
    REQUIRE(WaitFor([&] { return observer.packets >= before + 20; }));
}

TEST_CASE("Remote sessions can disconnect before negotiation completes",
          "[core][application][remote][stream]") {
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    REQUIRE(remote.createWebRtcSession("pending", "loopback") == Result::SUCCESS);
    remote.destroyWebRtcSession("pending");
    CHECK(remote.sessions.empty());

    Viewer viewer(stream);
    viewer.connect();
    REQUIRE(WaitFor([&] { return viewer.packets >= 20; }, 10s));
}

TEST_CASE("A vanished remote viewer does not interrupt other viewers",
          "[core][application][remote][stream]") {
    Stream stream;
    stream.start();
    Viewer observer(stream);
    Viewer vanished(stream);
    observer.connect();
    vanished.connect();
    REQUIRE(WaitFor([&] { return observer.packets >= 20; }, 10s));
    REQUIRE(WaitFor([&] { return vanished.packets >= 20; }, 10s));

    vanished.close();
    const unsigned before = observer.packets;
    REQUIRE(WaitFor([&] { return observer.packets >= before + 40; }));
    CHECK(stream.remote.findSession(vanished.sessionId));
}

TEST_CASE("Remote RTP delivery survives reconnects while other viewers keep receiving",
          "[core][application][remote][stream]") {
    const bool keepViewer = GENERATE(false, true);
    Stream stream;
    stream.start();
    std::optional<Viewer> observer;
    if (keepViewer) {
        observer.emplace(stream);
        observer->connect();
        REQUIRE(WaitFor([&] { return observer->packets >= 20; }, 10s));
    }
    for (unsigned i = 0; i < 8; ++i) {
        CAPTURE(keepViewer, i);
        {
            Viewer viewer(stream);
            viewer.connect();
            REQUIRE(WaitFor([&] { return viewer.packets >= 20; }, 10s));
        }
        if (keepViewer) {
            const unsigned before = observer->packets;
            REQUIRE(WaitFor([&] { return observer->packets >= before + 20; }));
        }
    }
}

#if defined(JST_OS_LINUX)
TEST_CASE("Keyring credentials and authenticated discovery coexist with WebRTC reconnects",
          "[core][application][remote][keyring][store]") {
    std::string enabled;
    if (Platform::EnvironmentVariable("CYBERETHER_TEST_SECRET_STORE", enabled) != Result::SUCCESS || enabled != "1") {
        SKIP("Set CYBERETHER_TEST_SECRET_STORE=1 inside an isolated keyring session");
    }

    Stream stream;
    stream.start();

    AgentTest::ProviderServer server;
    std::atomic<unsigned> authenticated = 0;
    server.server.Get("/models", [&](const auto& request, auto& response) {
        if (request.get_header_value("Authorization") != "Bearer keyring-webrtc-test-key") {
            response.status = 401;
            return;
        }
        ++authenticated;
        response.set_content(R"({"data":[{"id":"test-model"}]})", "application/json");
    });
    server.start();

    struct StoredKey {
        std::string id = "webrtc-test-" + Random::HexToken();
        ~StoredKey() {
            AgentProviders::Erase(id);
        }
    } key;
    REQUIRE(AgentProviders::Store(key.id, server.origin, "keyring-webrtc-test-key").success);
    const Settings::Intelligence::Provider provider{
        .name = "Keyring WebRTC test",
        .host = server.origin,
        .modelsPath = "/models",
        .apiKeyId = key.id,
    };
    const auto client = AgentProviders::Client(provider);

    for (unsigned connection = 0; connection < 2; ++connection) {
        CAPTURE(connection);
        std::vector<Agent::Provider::Model> models;
        Agent::Provider::Error error;
        REQUIRE(client->models(models, &error) == Result::SUCCESS);
        REQUIRE(models.size() == 1);
        CHECK(models.front().id == "test-model");
        CHECK(authenticated.load() == connection + 1);

        Viewer viewer(stream);
        viewer.connect();
        REQUIRE(WaitFor([&] { return viewer.packets >= 20; }, 10s));
    }

    REQUIRE(AgentProviders::Erase(key.id).success);
    std::string secret;
    CHECK(Platform::ReadSecret("CyberEther Intelligence", key.id, secret) == Result::ERROR);
    CHECK(AgentProviders::Erase(key.id).success);
}
#endif

TEST_CASE("Remote streams send RTCP sender reports",
          "[core][application][remote][stream]") {
    Stream stream;
    stream.start();
    Viewer viewer(stream);
    viewer.connect();
    REQUIRE(WaitFor([&] { return viewer.packets >= 20; }, 10s));
    REQUIRE(WaitFor([&] { return viewer.senderReport(); }, 10s));
}

TEST_CASE("Remote stream can stop and restart on the same instance",
          "[core][application][remote][stream]") {
    Stream stream;
    for (unsigned i = 0; i < 3; ++i) {
        CAPTURE(i);
        stream.start();
        {
            Viewer viewer(stream);
            viewer.connect();
            REQUIRE(WaitFor([&] { return viewer.packets >= 20; }, 10s));
        }
        stream.stop();
    }
}

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
TEST_CASE("Remote NVENC opens on the input CUDA context across stream restarts",
          "[core][application][remote][stream][cuda]") {
    SkipRemoteEncodingOnCI();
    if (!Backend::State<DeviceType::CUDA>()->isAvailable()) {
        SKIP("CUDA is unavailable");
    }
    if (!Instance::Remote::Encoder::Available({
            .codec = Instance::Remote::CodecType::H264,
            .encoder = Instance::Remote::EncoderType::NVENC,
            .device = DeviceType::CUDA,
            .context = Backend::State<DeviceType::CUDA>()->getContext(),
        })) {
        SKIP("NVENC is unavailable");
    }

    Stream stream;
    auto& remote = stream.remote;
    remote.size = {320, 240};
    remote.config.video.codec = Instance::Remote::CodecType::H264;
    remote.inputMemoryDevice_ = DeviceType::CUDA;
    remote.encoderType = Instance::Remote::EncoderType::NVENC;
    for (unsigned i = 0; i < 2; ++i) {
        CAPTURE(i);
        REQUIRE(remote.startStream() == Result::SUCCESS);
        CHECK(remote.encoder);
        REQUIRE(remote.stopStream() == Result::SUCCESS);
    }
}
#endif

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
TEST_CASE("Headless Metal frames reach a remote WebRTC viewer after restart",
          "[core][application][remote][stream][metal]") {
    SkipRemoteEncodingOnCI();
    const auto encoder = GENERATE(Instance::Remote::EncoderType::Software,
                                 Instance::Remote::EncoderType::VideoToolbox);
    struct Host {
        Instance instance;
        bool created = false;

        Host() {
            REQUIRE(instance.create({.headless = true,
                                     .size = {320, 240}, .framerate = 120}) == Result::SUCCESS);
            created = true;
            REQUIRE(instance.start() == Result::SUCCESS);
        }

        ~Host() {
            if (created) {
                instance.destroy();
            }
        }
    } host;
    std::shared_ptr<Viewport::Generic> viewport;
    REQUIRE(host.instance.viewportGet(viewport) == Result::SUCCESS);
    REQUIRE(viewport->device() == DeviceType::Metal);
    REQUIRE(host.instance.remote()->supported());
    const auto encoders = host.instance.remote()->available(Instance::Remote::CodecType::H264);
    if (std::find(encoders.begin(), encoders.end(), encoder) == encoders.end()) {
        SKIP("Requested H264 encoder is unavailable.");
    }

    Stream stream;
    auto& remote = stream.remote;
    remote.viewport = viewport.get();
    remote.viewportDevice = viewport->device();
    Instance::Remote::Config config;
    config.video.encoder = encoder;
    config.broker = {.address = "127.0.0.1", .port = 0};

    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        CAPTURE(attempt, encoder);
        REQUIRE(remote.create(config) == Result::SUCCESS);
        {
            Viewer peer(stream);
            peer.connect();
            const auto deadline = std::chrono::steady_clock::now() + 10s;
            while (peer.packets < 3 && std::chrono::steady_clock::now() < deadline) {
                REQUIRE(host.instance.present() == Result::SUCCESS);
                REQUIRE(remote.captureFrame(true) == Result::SUCCESS);
            }
            REQUIRE(peer.packets >= 3);
        }
        REQUIRE(remote.destroy() == Result::SUCCESS);
    }

    const auto ownedRemote = host.instance.remote();
    REQUIRE(ownedRemote->create(config) == Result::SUCCESS);
    for (unsigned i = 0; i < 4; ++i) {
        REQUIRE(host.instance.present() == Result::SUCCESS);
    }
    REQUIRE(host.instance.destroy() == Result::SUCCESS);
    host.created = false;
    REQUIRE_FALSE(ownedRemote->started());
}
#endif

TEST_CASE("Remote broker rejects incomplete or unreadable TLS configuration",
          "[core][application][remote][broker]") {
    Instance::Remote::Broker server;
    CHECK(server.create({.address = "127.0.0.1", .port = 0, .certificate = "cert.pem"}) == Result::ERROR);
    CHECK(server.create({.address = "127.0.0.1", .port = 0, .key = "key.pem"}) == Result::ERROR);
    CHECK(server.create({.address = "127.0.0.1", .port = 0,
                            .certificate = "missing-cert.pem", .key = "missing-key.pem"}) == Result::ERROR);
    CHECK(server.create({.address = "127.0.0.1", .port = 0, .publicUrl = "ws://example.com"}) == Result::ERROR);
    CHECK(server.origin().empty());
}

TEST_CASE("Remote broker admits, relays, and ends a viewer session",
          "[core][application][remote][broker]") {
    Instance::Remote::Broker server;
    REQUIRE(server.create({.address = "127.0.0.1", .port = 0}) == Result::SUCCESS);
    REQUIRE_FALSE(server.secure());
    REQUIRE(server.origin().starts_with("http://127.0.0.1:"));
    REQUIRE(server.publicOrigin() == server.origin());

    httplib::Client http(server.origin());
    http.set_decompress(false);
    const auto page = http.Get("/remote");
    REQUIRE(page);
    CHECK(page->status == 200);
    CHECK(page->get_header_value("Content-Type").starts_with("text/html"));
    CHECK(page->get_header_value("Content-Encoding") == "gzip");
    REQUIRE(page->body.size() == Resources::RemoteViewerPageGzipSize);
    CHECK(static_cast<unsigned char>(page->body[0]) == 0x1f);
    CHECK(static_cast<unsigned char>(page->body[1]) == 0x8b);

    const httplib::Headers producerHeaders = {{"Authorization", "Bearer " + server.producerKey()}};

    BrokerPeer anonymous(server.origin());
    const auto unauthorized = anonymous.connect();
    CHECK_FALSE(unauthorized);
    CHECK(unauthorized.status() == 401);

    BrokerPeer forged(server.origin(), "", {{"Authorization", "Bearer wrong"}});
    const auto forbidden = forged.connect();
    CHECK_FALSE(forbidden);
    CHECK(forbidden.status() == 401);

    BrokerPeer producer(server.origin(), "", producerHeaders);
    REQUIRE(producer.connect());
    const auto welcome = producer.next("welcome");
    REQUIRE(welcome["peerId"].is_string());
    const std::string producerId = welcome["peerId"];

    producer.send({{"type", "createRoom"}});
    const auto room = producer.next("roomCreated");
    const std::string roomId = room["roomId"];
    const std::string consumerToken = room["consumerToken"];
    CHECK(room["clientDomain"] == server.publicOrigin() + "/remote");
    CHECK_FALSE(room.contains("producerToken"));
    CHECK(producer.next("roomState")["waiting"].empty());

    producer.send({{"type", "ping"}});
    producer.next("pong");

    producer.send({{"type", "createRoom"}});
    CHECK(producer.next("error")["code"] == "already_authenticated");

    BrokerPeer stranger(server.origin(), "?tokenHash=" + BrokerTokenHash("wrong"));
    const auto rejected = stranger.connect();
    CHECK_FALSE(rejected);
    CHECK(rejected.status() == 404);

    BrokerPeer impostor(server.origin(), "", producerHeaders);
    REQUIRE(impostor.connect());
    impostor.next("welcome");
    impostor.send({{"type", "createRoom"}});
    CHECK(impostor.next("error")["code"] == "already_authenticated");

    BrokerPeer viewer(server.origin(), "?tokenHash=" + BrokerTokenHash(consumerToken));
    REQUIRE(viewer.connect());
    const std::string viewerId = viewer.next("welcome")["peerId"];

    viewer.send({{"type", "setPeerStatus"}, {"roles", {"listener"}}, {"meta", {{"token", "wrong"}}}});
    CHECK(viewer.next("error")["code"] == "invalid_token");

    viewer.send({{"type", "setPeerStatus"}, {"roles", {"listener"}}, {"meta", {{"token", consumerToken}}}});
    const std::string sessionId = viewer.next("sessionRegistered")["sessionId"];
    const auto waiting = producer.next("roomState");
    REQUIRE(waiting["waiting"] == nlohmann::json::array({sessionId}));
    CHECK(waiting["active"].empty());

    viewer.send({{"type", "peer"}, {"sessionId", sessionId}, {"ice", {{"candidate", "early"}}}});
    CHECK(viewer.next("error")["code"] == "session_not_started");

    viewer.send({{"type", "approveSession"}, {"sessionId", sessionId}});
    CHECK(viewer.next("error")["code"] == "unauthorized");

    producer.send({{"type", "approveSession"}, {"sessionId", sessionId}});
    CHECK(producer.next("roomState")["active"] == nlohmann::json::array({sessionId}));
    const auto start = producer.next("startSession");
    CHECK(start["sessionId"] == sessionId);
    CHECK(start["peerId"] == viewerId);
    CHECK(start["offer"].is_null());
    const auto started = viewer.next("sessionStarted");
    CHECK(started["sessionId"] == sessionId);
    CHECK(started["peerId"] == producerId);

    producer.send({{"type", "peer"}, {"sessionId", sessionId},
                   {"sdp", {{"type", "offer"}, {"sdp", "v=0"}, {"extra", true}}}});
    const auto offer = viewer.next("peer");
    CHECK(offer["sdp"] == nlohmann::json({{"type", "offer"}, {"sdp", "v=0"}}));

    viewer.send({{"type", "peer"}, {"sessionId", sessionId},
                 {"ice", {{"candidate", "candidate:1 1 tcp 1 127.0.0.1 9 typ host tcptype passive"},
                          {"sdpMLineIndex", 0}}}});
    const auto candidate = producer.next("peer");
    CHECK(candidate["ice"]["candidate"] == "candidate:1 1 tcp 1 127.0.0.1 9 typ host tcptype passive");
    CHECK(candidate["ice"]["sdpMLineIndex"] == 0);

    viewer.send({{"type", "peer"}, {"sessionId", sessionId},
                 {"sdp", {{"type", "answer"}, {"sdp", "v=0"}}}, {"ice", {{"candidate", ""}}}});
    CHECK(viewer.next("error")["code"] == "invalid_message");

    viewer.send({{"type", "endSession"}, {"sessionId", sessionId}});
    CHECK(viewer.next("endSession")["sessionId"] == sessionId);
    CHECK(producer.next("roomState")["active"].empty());
    CHECK(producer.next("endSession")["sessionId"] == sessionId);
    CHECK(viewer.closed());

    BrokerPeer pending(server.origin(), "?tokenHash=" + BrokerTokenHash(consumerToken));
    REQUIRE(pending.connect());
    pending.send({{"type", "setPeerStatus"}, {"roles", {"listener"}}, {"meta", {{"token", consumerToken}}}});
    const std::string pendingId = pending.next("sessionRegistered")["sessionId"];

    producer.client->close();
    CHECK(pending.next("endSession")["sessionId"] == pendingId);
    CHECK(pending.closed());
    BrokerPeer late(server.origin(), "?tokenHash=" + BrokerTokenHash(consumerToken));
    const auto gone = late.connect();
    CHECK_FALSE(gone);
    CHECK(gone.status() == 404);

    BrokerPeer next(server.origin(), "", producerHeaders);
    REQUIRE(next.connect());
    next.next("welcome");
    next.send({{"type", "createRoom"}});
    CHECK(next.next("roomCreated")["roomId"] != roomId);

    REQUIRE(server.destroy() == Result::SUCCESS);
    CHECK(next.closed());
}

TEST_CASE("Remote hosts approve and remove viewer sessions by their full id",
          "[core][application][remote][broker]") {
    Stream stream;
    auto& remote = stream.remote;
    remote.config.broker = {.address = "127.0.0.1", .port = 0};
    remote.size = {320, 240};
    remote.inputMemoryDevice_ = DeviceType::CPU;
    remote.encoderType = Instance::Remote::EncoderType::Software;
    REQUIRE(remote.createBroker() == Result::SUCCESS);
    remote.started_ = true;

    const auto join = [&](BrokerPeer& viewer) {
        REQUIRE(viewer.connect());
        viewer.send({{"type", "setPeerStatus"}, {"roles", {"listener"}}, {"meta", {{"token", remote.consumerToken}}}});
        return viewer.next("sessionRegistered")["sessionId"].get<std::string>();
    };
    const auto roster = [&](std::size_t active, std::vector<std::string> waiting) {
        return WaitFor([&] {
            std::lock_guard<std::mutex> lock(remote.remoteStateMutex);
            return remote.clients_.size() == active && remote.waitlist_ == waiting;
        });
    };

    BrokerPeer first(remote.broker->origin(), "?tokenHash=" + BrokerTokenHash(remote.consumerToken));
    BrokerPeer second(remote.broker->origin(), "?tokenHash=" + BrokerTokenHash(remote.consumerToken));
    const std::string firstId = join(first);
    REQUIRE(roster(0, {firstId}));
    const std::string secondId = join(second);
    REQUIRE(roster(0, {firstId, secondId}));

    CHECK(remote.approveClient("") == Result::ERROR);
    CHECK(remote.approveClient(firstId.substr(firstId.size() - 6)) == Result::ERROR);
    CHECK(remote.removeClient("") == Result::ERROR);
    CHECK(remote.removeClient("unknown-session") == Result::ERROR);

    REQUIRE(remote.approveClient(firstId) == Result::SUCCESS);
    CHECK(first.next("sessionStarted")["sessionId"] == firstId);
    REQUIRE(roster(1, {secondId}));
    {
        std::lock_guard<std::mutex> lock(remote.remoteStateMutex);
        CHECK(remote.clients_.front().sessionId == firstId);
    }
    CHECK(remote.approveClient(firstId) == Result::ERROR);

    REQUIRE(remote.removeClient(secondId) == Result::SUCCESS);
    CHECK(second.next("endSession")["sessionId"] == secondId);
    REQUIRE(roster(1, {}));

    REQUIRE(remote.removeClient(firstId) == Result::SUCCESS);
    CHECK(first.next("endSession")["sessionId"] == firstId);
    REQUIRE(roster(0, {}));

    remote.started_ = false;
    CHECK(remote.approveClient(firstId) == Result::ERROR);
    CHECK(remote.removeClient(firstId) == Result::ERROR);
    REQUIRE(remote.destroy() == Result::SUCCESS);
}

TEST_CASE("Remote streaming hosts its own broker",
          "[core][application][remote][broker]") {
    Stream stream;
    auto& remote = stream.remote;
    remote.config.broker = {.address = "127.0.0.1", .port = 0};
    remote.size = {320, 240};
    remote.inputMemoryDevice_ = DeviceType::CPU;
    remote.encoderType = Instance::Remote::EncoderType::Software;

    REQUIRE(remote.createBroker() == Result::SUCCESS);
    REQUIRE(remote.broker);
    CHECK(remote.inviteUrl_ == remote.broker->publicOrigin() + "/remote#" + remote.consumerToken);
    CHECK(remote.consumerToken.size() == 43);

    REQUIRE(remote.destroy() == Result::SUCCESS);
    CHECK_FALSE(remote.broker);
    CHECK(remote.inviteUrl_.empty());
}

TEST_CASE("Remote streaming reaches its TLS broker over loopback",
          "[core][application][remote][broker]") {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("cyberether-broker-tls-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    WriteSelfSignedCertificate(directory / "cert.pem", directory / "key.pem");

    Stream stream;
    auto& remote = stream.remote;
    remote.config.broker = {
        .address = "127.0.0.1",
        .port = 0,
        .publicUrl = "https://viewer.example:8443/",
        .certificate = (directory / "cert.pem").string(),
        .key = (directory / "key.pem").string(),
    };
    remote.size = {320, 240};
    remote.inputMemoryDevice_ = DeviceType::CPU;
    remote.encoderType = Instance::Remote::EncoderType::Software;

    REQUIRE(remote.createBroker() == Result::SUCCESS);
    REQUIRE(remote.broker);
    CHECK(remote.broker->secure());
    CHECK(remote.broker->origin().starts_with("https://127.0.0.1:"));
    CHECK(remote.inviteUrl_ == "https://viewer.example:8443/remote#" + remote.consumerToken);

    REQUIRE(remote.destroy() == Result::SUCCESS);
    CHECK_FALSE(remote.broker);
    std::filesystem::remove_all(directory);
}

TEST_CASE("Remote streaming starts with the broker bound to a network address",
          "[core][application][remote][broker]") {
    std::string address;
    if (Platform::LocalNetworkAddress(address) != Result::SUCCESS) {
        SKIP("No local network address is available.");
    }

    Stream stream;
    auto& remote = stream.remote;
    remote.config.broker = {.address = address, .port = 0};
    remote.size = {320, 240};
    remote.inputMemoryDevice_ = DeviceType::CPU;
    remote.encoderType = Instance::Remote::EncoderType::Software;

    REQUIRE(remote.createBroker() == Result::SUCCESS);
    CHECK(remote.broker->origin().starts_with("http://" + address + ":"));
    CHECK(remote.inviteUrl_.starts_with("http://" + address + ":"));
    REQUIRE(remote.destroy() == Result::SUCCESS);
}

TEST_CASE("Remote broker shutdown interrupts peers that keep the socket busy",
          "[core][application][remote][broker]") {
    Instance::Remote::Broker server;
    REQUIRE(server.create({.address = "127.0.0.1", .port = 0}) == Result::SUCCESS);

    BrokerPeer producer(server.origin(), "", {{"Authorization", "Bearer " + server.producerKey()}});
    producer.client->set_websocket_ping_interval(1);
    REQUIRE(producer.connect());
    producer.next("welcome");

    std::atomic<bool> chatting = true;
    std::thread chatter([&] {
        while (chatting) {
            producer.client->send(R"({"type":"ping"})");
            std::this_thread::sleep_for(50ms);
        }
    });

    std::this_thread::sleep_for(200ms);
    const auto started = std::chrono::steady_clock::now();
    REQUIRE(server.destroy() == Result::SUCCESS);
    CHECK(std::chrono::steady_clock::now() - started < 3s);

    chatting = false;
    chatter.join();
}

TEST_CASE("Remote media address limits candidates to that address",
          "[core][application][remote][media]") {
    Stream stream;
    stream.remote.config.media.address = "127.0.0.1";
    stream.start();

    Viewer viewer(stream);
    viewer.connect();
    REQUIRE(WaitFor([&] { return viewer.packets > 0; }, 10s));

    std::lock_guard<std::mutex> lock(viewer.mutex);
    REQUIRE_FALSE(viewer.candidates.empty());
    for (const auto& candidate : viewer.candidates) {
        CAPTURE(candidate);
        std::string lower = candidate;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        CHECK(lower.find(" 127.0.0.1 ") != std::string::npos);
        CHECK(lower.find(" udp ") != std::string::npos);
    }
}

TEST_CASE("Remote rejects a media address that is not an IP address",
          "[core][application][remote][media]") {
    Instance::Remote remote(nullptr);
    Instance::Remote::Config config;
    config.media.address = "localhost";
    JST_LOG_LAST_ERROR().clear();
    REQUIRE(remote.create(config) == Result::ERROR);
    REQUIRE(JST_LOG_LAST_ERROR() == "[REMOTE] Invalid media address 'localhost'. Expected an IP address.");
}
