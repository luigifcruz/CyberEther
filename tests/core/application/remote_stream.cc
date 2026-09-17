#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>

#include "../../../src/instance_remote_impl.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/backend/base.hh"
#endif

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtp/gstrtcpbuffer.h>
#include <gst/video/video-info.h>

using namespace Jetstream;
using namespace std::chrono_literals;

extern "C" void gst_init_static_plugins();

namespace {

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

    void start(bool produceFrames = true) {
        gst_init(nullptr, nullptr);
        gst_init_static_plugins();
        remote.size = {320, 240};
        remote.config.codec = Instance::Remote::CodecType::H264;
        remote.inputMemoryDevice_ = DeviceType::CPU;
        remote.encodingStrategy = Instance::Remote::Impl::EncodingStrategyType::Software;
        REQUIRE(remote.startStream() == Result::SUCCESS);
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

struct Broker {
    httplib::Server server;
    std::thread listener;
    std::string origin;
    std::atomic<unsigned> rooms = 0;

    Broker() {
        server.WebSocket("/api/v1/remote/signaller", [this](const httplib::Request&, httplib::ws::WebSocket& ws) {
            ws.send(R"({"type":"welcome","peerId":"producer"})");
            std::string payload;
            while (ws.read(payload) == httplib::ws::Text) {
                const auto message = nlohmann::json::parse(payload);
                if (message.value("type", "") == "disconnect") {
                    ws.close();
                    return;
                }
                if (message.value("type", "") == "createRoom") {
                    const auto room = std::to_string(++rooms);
                    ws.send(nlohmann::json{{"type", "roomCreated"}, {"roomId", room},
                                          {"consumerToken", "token-" + room},
                                          {"clientDomain", origin + "/remote"}}.dump());
                    ws.send(R"({"type":"roomState","waiting":["waiting"],"active":["viewer"]})");
                }
            }
        });
        const int port = server.bind_to_any_port("127.0.0.1");
        REQUIRE(port > 0);
        origin = "http://127.0.0.1:" + std::to_string(port);
        listener = std::thread([this] { server.listen_after_bind(); });
    }

    ~Broker() {
        server.stop();
        if (listener.joinable()) listener.join();
    }
};

void Loopback(GstElement* webrtc) {
    GstWebRTCICE* ice = nullptr;
    g_object_get(webrtc, "ice-agent", &ice, nullptr);
    gboolean added = FALSE;
    g_signal_emit_by_name(ice, "add-local-ip-address", "127.0.0.1", &added);
    gst_object_unref(ice);
    REQUIRE(added);
}

void SetDescription(GstElement* webrtc, const char* action, GstWebRTCSessionDescription* desc) {
    GstPromise* promise = gst_promise_new();
    g_signal_emit_by_name(webrtc, action, desc, promise);
    const auto result = gst_promise_wait(promise);
    gst_promise_unref(promise);
    REQUIRE(result == GST_PROMISE_RESULT_REPLIED);
}

GstWebRTCSessionDescription* CreateDescription(GstElement* webrtc, bool offer) {
    GstPromise* promise = gst_promise_new();
    g_signal_emit_by_name(webrtc, offer ? "create-offer" : "create-answer", nullptr, promise);
    REQUIRE(gst_promise_wait(promise) == GST_PROMISE_RESULT_REPLIED);
    GstWebRTCSessionDescription* desc = nullptr;
    gst_structure_get(gst_promise_get_reply(promise), offer ? "offer" : "answer",
                      GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &desc, nullptr);
    gst_promise_unref(promise);
    REQUIRE(desc);
    return desc;
}

void Negotiate(GstElement* from, GstElement* to, bool offer) {
    GstWebRTCSessionDescription* desc = CreateDescription(from, offer);
    SetDescription(from, "set-local-description", desc);
    SetDescription(to, "set-remote-description", desc);
    gst_webrtc_session_description_free(desc);
}

struct Peer {
    Instance::Remote::Impl& remote;
    std::string id;
    GstElement* pipeline = nullptr;
    GstElement* receiver = nullptr;
    GstElement* sender = nullptr;
    GstElement* sink = nullptr;
    gulong senderIce = 0;
    gulong receiverIce = 0;
    std::atomic<unsigned> packets = 0;
    std::atomic<unsigned> senderReports = 0;

    Peer(Stream& stream, std::string id) : remote(stream.remote), id(std::move(id)) {}

    void prepare() {
        REQUIRE(remote.createWebRtcSession(id, "loopback") == Result::SUCCESS);
        auto& session = *remote.sessions.at(id);
        sender = GST_ELEMENT(gst_object_ref(session.webrtc));
        g_signal_handler_disconnect(sender, session.iceHandler);
        session.iceHandler = 0;
        Loopback(sender);
        pipeline = gst_pipeline_new(nullptr);
        receiver = gst_element_factory_make("webrtcbin", nullptr);
        sink = gst_element_factory_make("appsink", nullptr);
        REQUIRE(pipeline);
        REQUIRE(receiver);
        REQUIRE(sink);
        gst_bin_add_many(GST_BIN(pipeline), receiver, sink, nullptr);
        g_object_set(receiver, "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, nullptr);
        Loopback(receiver);
        g_object_set(sink, "sync", FALSE, "async", FALSE, "emit-signals", TRUE,
                     "enable-last-sample", FALSE, "wait-on-eos", FALSE, nullptr);
        g_signal_connect(receiver, "pad-added", G_CALLBACK(+[](GstElement*, GstPad* pad, gpointer data) {
            if (GST_PAD_DIRECTION(pad) != GST_PAD_SRC) return;
            auto* self = static_cast<Peer*>(data);
            GstPad* target = gst_element_get_static_pad(self->sink, "sink");
            gst_pad_link(pad, target);
            gst_object_unref(target);
        }), this);
        g_signal_connect(sink, "new-sample", G_CALLBACK(+[](GstAppSink* sink, gpointer data) -> GstFlowReturn {
            GstSample* sample = gst_app_sink_pull_sample(sink);
            if (sample) {
                ++static_cast<Peer*>(data)->packets;
                gst_sample_unref(sample);
            }
            return GST_FLOW_OK;
        }), this);
        auto ice = +[](GstElement*, guint index, gchar* candidate, gpointer other) {
            g_signal_emit_by_name(other, "add-ice-candidate", index, candidate);
        };
        senderIce = g_signal_connect(sender, "on-ice-candidate", G_CALLBACK(ice), receiver);
        receiverIce = g_signal_connect(receiver, "on-ice-candidate", G_CALLBACK(ice), sender);
        REQUIRE(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
    }

    void connect() {
        prepare();
        negotiate();
    }

    void negotiate() {
        remote.createControlChannel(id);
        Negotiate(sender, receiver, true);
        Negotiate(receiver, sender, false);

        GstElement* rtpbin = gst_bin_get_by_name(GST_BIN(receiver), "rtpbin");
        REQUIRE(rtpbin);
        GstPad* rtcp = gst_element_get_static_pad(rtpbin, "recv_rtcp_sink_0");
        gst_object_unref(rtpbin);
        REQUIRE(rtcp);
        gst_pad_add_probe(rtcp, GST_PAD_PROBE_TYPE_BUFFER, +[](GstPad*, GstPadProbeInfo* info, gpointer data) {
            GstRTCPBuffer buffer = GST_RTCP_BUFFER_INIT;
            if (gst_rtcp_buffer_map(GST_PAD_PROBE_INFO_BUFFER(info), GST_MAP_READ, &buffer)) {
                GstRTCPPacket packet;
                if (gst_rtcp_buffer_get_first_packet(&buffer, &packet)) {
                    do {
                        if (gst_rtcp_packet_get_type(&packet) == GST_RTCP_TYPE_SR)
                            ++static_cast<Peer*>(data)->senderReports;
                    } while (gst_rtcp_packet_move_to_next(&packet));
                }
                gst_rtcp_buffer_unmap(&buffer);
            }
            return GST_PAD_PROBE_OK;
        }, this, nullptr);
        gst_object_unref(rtcp);
    }

    ~Peer() {
        if (senderIce) g_signal_handler_disconnect(sender, senderIce);
        if (receiverIce) g_signal_handler_disconnect(receiver, receiverIce);
        remote.destroyWebRtcSession(id);
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
        }
        gst_clear_object(&sender);
    }
};

struct PadProbe {
    GstPad* pad = nullptr;
    gulong id = 0;

    void remove() {
        if (id) gst_pad_remove_probe(pad, id);
        id = 0;
    }

    ~PadProbe() {
        remove();
        gst_clear_object(&pad);
    }
};

struct InputContext {
    ImGuiContext* context = ImGui::CreateContext();

    InputContext() {
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = {320.0f, 240.0f};
        io.ConfigInputTrickleEventQueue = false;
        io.Fonts->AddFontDefault();
    }

    ~InputContext() { ImGui::DestroyContext(context); }

    template<typename Check>
    void frame(Instance::Remote::Impl& remote, Check check) {
        REQUIRE(remote.processInput() == Result::SUCCESS);
        ImGui::NewFrame();
        check(ImGui::GetIO());
        ImGui::EndFrame();
    }
};

void SendInput(Instance::Remote::Impl& remote, const std::string& sessionId, const nlohmann::json& input) {
    Instance::Remote::Impl::ControlChannelContext context{&remote, sessionId};
    auto payload = input.dump();
    Instance::Remote::Impl::onMessageCallback(nullptr, payload.data(), &context);
}

}  // namespace

TEST_CASE("Only the first remote consumer can control the interface",
          "[core][application][remote][input]") {
    const std::string sender = GENERATE("z-first", "a-second", "unknown");
    const bool controls = sender == "z-first";
    const bool macOS = GENERATE(false, true);
    InputContext input;
    ImGui::GetIO().ConfigMacOSXBehaviors = macOS;
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);
    REQUIRE(remote.createWebRtcSession("a-second", "second") == Result::SUCCESS);

    SendInput(remote, sender, {{"kind", "keyboard"}, {"action", "down"}, {"code", "KeyA"}, {"ctrlKey", true}});
    SendInput(remote, sender, {{"kind", "mouse"}, {"act", "down"}, {"button", 0}, {"x", 0.5}, {"y", 0.5}});
    SendInput(remote, sender, {{"kind", "wheel"}, {"deltaY", -100}});
    SendInput(remote, sender, {{"kind", "text"}, {"text", "x"}});
    input.frame(remote, [&](const ImGuiIO& io) {
        CHECK(ImGui::IsKeyDown(ImGuiKey_A) == controls);
        // ImGui swaps Ctrl/Super and maps Ctrl+left click to right click on macOS.
        CHECK(io.KeyCtrl == (controls && !macOS));
        CHECK(io.KeySuper == (controls && macOS));
        CHECK(io.MouseDown[0] == (controls && !macOS));
        CHECK(io.MouseDown[1] == (controls && macOS));
        CHECK(io.MouseWheel == (controls ? 1.0f : 0.0f));
        CHECK(io.InputQueueCharacters.Size == (controls ? 1 : 0));
        CHECK((io.MousePos.x == 160.0f) == controls);
    });

    Instance::Remote::Impl::ControlChannelContext context{&remote, sender};
    Instance::Remote::Impl::onChannelClosedCallback(nullptr, &context);
    input.frame(remote, [](const ImGuiIO& io) {
        CHECK_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
        CHECK_FALSE(io.KeyCtrl);
        CHECK_FALSE(io.KeySuper);
        CHECK_FALSE(io.MouseDown[0]);
        CHECK_FALSE(io.MouseDown[1]);
    });
}

TEST_CASE("Remote control follows connection order and releases input on handoff",
          "[core][application][remote][input]") {
    InputContext input;
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);
    REQUIRE(remote.createWebRtcSession("m-second", "second") == Result::SUCCESS);
    REQUIRE(remote.createWebRtcSession("a-third", "third") == Result::SUCCESS);
    // Duplicate start messages must not create another position in the queue.
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);

    SendInput(remote, "z-first", {{"kind", "keyboard"}, {"action", "down"}, {"code", "KeyA"}, {"shiftKey", true}});
    SendInput(remote, "z-first", {{"kind", "mouse"}, {"act", "down"}, {"button", 0}, {"x", 0.5}, {"y", 0.5}});
    input.frame(remote, [](const ImGuiIO& io) {
        CHECK(ImGui::IsKeyDown(ImGuiKey_A));
        CHECK(io.KeyShift);
        CHECK(io.MouseDown[0]);
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
    input.frame(remote, [](const ImGuiIO& io) {
        CHECK_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
        CHECK(ImGui::IsKeyDown(ImGuiKey_B));
        CHECK_FALSE(io.KeyShift);
        CHECK_FALSE(io.MouseDown[0]);
        CHECK(io.MousePos.x == 160.0f);
        CHECK(io.InputQueueCharacters.Size == 1);
        if (io.InputQueueCharacters.Size == 1) CHECK(io.InputQueueCharacters[0] == 'b');
    });

    // A reconnect joins at the back, even if it was the original controller.
    REQUIRE(remote.createWebRtcSession("z-first", "first") == Result::SUCCESS);
    remote.destroyWebRtcSession("m-second");
    SendInput(remote, "z-first", {{"kind", "text"}, {"text", "reconnected"}});
    SendInput(remote, "a-third", {{"kind", "text"}, {"text", "c"}});
    input.frame(remote, [](const ImGuiIO& io) {
        CHECK_FALSE(ImGui::IsKeyDown(ImGuiKey_B));
        CHECK(io.InputQueueCharacters.Size == 1);
        if (io.InputQueueCharacters.Size == 1) CHECK(io.InputQueueCharacters[0] == 'c');
    });

    // Removing a spectator cannot release the controller's held input.
    SendInput(remote, "a-third", {{"kind", "mouse"}, {"act", "down"}, {"button", 0}});
    remote.destroyWebRtcSession("z-first");
    input.frame(remote, [](const ImGuiIO& io) { CHECK(io.MouseDown[0]); });

    stream.stop();
    stream.start();
    REQUIRE(remote.createWebRtcSession("new-first", "new") == Result::SUCCESS);
    SendInput(remote, "new-first", {{"kind", "text"}, {"text", "d"}});
    input.frame(remote, [](const ImGuiIO& io) {
        CHECK_FALSE(io.MouseDown[0]);
        CHECK(io.InputQueueCharacters.Size == 1);
        if (io.InputQueueCharacters.Size == 1) CHECK(io.InputQueueCharacters[0] == 'd');
    });
}

TEST_CASE("Broker disconnect stops remote streaming and releases its room and input state",
          "[core][application][remote][broker]") {
    InputContext input;
    Broker broker;
    REQUIRE(WaitFor([&] { return broker.server.is_running(); }));
    Stream stream;
    auto& remote = stream.remote;
    gst_init(nullptr, nullptr);
    gst_init_static_plugins();
    remote.config.broker = broker.origin;
    remote.size = {320, 240};

    for (unsigned i = 1; i <= 2; ++i) {
        CAPTURE(i);
        remote.inputMemoryDevice_ = DeviceType::CPU;
        remote.encodingStrategy = Instance::Remote::Impl::EncodingStrategyType::Software;
        REQUIRE(remote.createBroker() == Result::SUCCESS);
        remote.started_ = true;
        REQUIRE(remote.roomId_ == std::to_string(i));
        REQUIRE(WaitFor([&] {
            std::lock_guard<std::mutex> lock(remote.remoteStateMutex);
            return remote.clients_.size() == 1 && remote.waitlist_.size() == 1;
        }));
        REQUIRE(remote.createWebRtcSession("viewer", "loopback") == Result::SUCCESS);
        SendInput(remote, "viewer", {{"kind", "keyboard"}, {"action", "down"},
                                      {"code", "KeyA"}, {"shiftKey", true}});
        SendInput(remote, "viewer", {{"kind", "mouse"}, {"act", "down"}, {"button", 0}});
        input.frame(remote, [](const ImGuiIO& io) {
            CHECK(ImGui::IsKeyDown(ImGuiKey_A));
            CHECK(io.KeyShift);
            CHECK(io.MouseDown[0]);
        });
        REQUIRE(remote.captureFrame() == Result::SUCCESS);
        REQUIRE(remote.started_);
        REQUIRE(remote.sendSignallerMessage({{"type", "disconnect"}}));
        REQUIRE(WaitFor([&] { return !remote.signallerRunning; }));
        REQUIRE(remote.captureFrame() == Result::SUCCESS);

        CHECK_FALSE(remote.started_);
        CHECK_FALSE(remote.streaming);
        CHECK_FALSE(remote.pipeline);
        CHECK_FALSE(remote.signallerClient);
        CHECK_FALSE(remote.signallerThread.joinable());
        CHECK(remote.sessions.empty());
        CHECK(remote.roomId_.empty());
        CHECK(remote.consumerToken.empty());
        CHECK(remote.inviteUrl_.empty());
        CHECK(remote.clients_.empty());
        CHECK(remote.waitlist_.empty());
        CHECK(remote.inputSessionOrder.empty());
        input.frame(remote, [](const ImGuiIO& io) {
            CHECK_FALSE(ImGui::IsKeyDown(ImGuiKey_A));
            CHECK_FALSE(io.KeyShift);
            CHECK_FALSE(io.MouseDown[0]);
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
    remote.config.framerate = captureFps;
    auto capture = std::make_unique<Capture>();
    auto* counter = capture.get();
    remote.frameCapture = std::move(capture);
    const auto start = std::chrono::steady_clock::time_point(1s);
    for (U32 frame = 0; frame < renderFps * 2; ++frame) {
        const auto now = start + std::chrono::nanoseconds(frame * GST_SECOND / renderFps);
        REQUIRE(remote.captureFrame(now) == Result::SUCCESS);
    }
    CHECK(counter->frames == std::min(renderFps, captureFps) * 2);

    const auto before = counter->frames;
    const auto resumed = start + 10s;
    REQUIRE(remote.captureFrame(resumed) == Result::SUCCESS);
    for (unsigned i = 0; i < 8; ++i) {
        REQUIRE(remote.captureFrame(resumed + 1ms) == Result::SUCCESS);
    }
    CHECK(counter->frames == before + 1);
    REQUIRE(remote.captureFrame(resumed + std::chrono::nanoseconds(GST_SECOND / captureFps)) == Result::SUCCESS);
    CHECK(counter->frames == before + 2);
}

TEST_CASE("Remote CPU encoding preserves the configured frame rate and source colorimetry",
          "[core][application][remote][stream][caps]") {
    const U32 framerate = GENERATE(30u, 60u);
    CAPTURE(framerate);
    Stream stream;
    stream.remote.config.framerate = framerate;
    stream.start();
    Peer viewer(stream, "viewer");
    viewer.connect();
    REQUIRE(WaitFor([&] { return viewer.packets >= 20; }));

    const auto videoInfo = [&](const char* elementName, const char* padName) {
        CAPTURE(elementName, padName);
        GstElement* element = gst_bin_get_by_name(GST_BIN(stream.remote.pipeline), elementName);
        REQUIRE(element);
        GstPad* pad = gst_element_get_static_pad(element, padName);
        gst_object_unref(element);
        REQUIRE(pad);
        GstCaps* caps = gst_pad_get_current_caps(pad);
        gst_object_unref(pad);
        REQUIRE(caps);
        GstVideoInfo info;
        const bool parsed = gst_video_info_from_caps(&info, caps);
        gst_caps_unref(caps);
        REQUIRE(parsed);
        CHECK(GST_VIDEO_INFO_FPS_N(&info) == static_cast<gint>(framerate));
        CHECK(GST_VIDEO_INFO_FPS_D(&info) == 1);
        return info;
    };

    const auto raw = videoInfo("rawparser", "src");
    CHECK(GST_VIDEO_INFO_FORMAT(&raw) == GST_VIDEO_FORMAT_BGRA);
    CHECK(GST_VIDEO_INFO_WIDTH(&raw) == static_cast<gint>(stream.remote.size.x));
    CHECK(GST_VIDEO_INFO_HEIGHT(&raw) == static_cast<gint>(stream.remote.size.y));
    CHECK(gst_video_colorimetry_matches(&raw.colorimetry, GST_VIDEO_COLORIMETRY_SRGB));

    const auto encodedInput = videoInfo("encoder", "sink");
    CHECK(encodedInput.colorimetry.transfer == GST_VIDEO_TRANSFER_SRGB);
    CHECK(encodedInput.colorimetry.primaries == GST_VIDEO_COLOR_PRIMARIES_BT709);
}

TEST_CASE("Remote offers wait for encoded H264 caps and advertise the actual profile and level",
          "[core][application][remote][stream][caps]") {
    const bool capsFirst = GENERATE(false, true);
    CAPTURE(capsFirst);
    Stream stream;
    stream.start(false);
    auto& remote = stream.remote;

    // Disconnecting while waiting for the first frame must cancel that offer,
    // including when a subsequent session reuses the same identifier.
    remote.handleStartSession({{"sessionId", "viewer"}, {"peerId", "cancelled"}});
    remote.destroyWebRtcSession("viewer");

    Peer viewer(stream, "viewer");
    viewer.prepare();
    auto& session = *remote.sessions.at(viewer.id);
    std::unique_ptr<GstCaps, decltype(&gst_caps_unref)> caps(nullptr, gst_caps_unref);
    const auto readCaps = [&] {
        caps.reset(gst_pad_get_current_caps(session.webrtcSinkPad));
        return caps != nullptr;
    };
    std::unique_ptr<GstWebRTCSessionDescription, decltype(&gst_webrtc_session_description_free)>
        offer(nullptr, gst_webrtc_session_description_free);
    const auto readOffer = [&] {
        GstWebRTCSessionDescription* desc = nullptr;
        g_object_get(viewer.sender, "local-description", &desc, nullptr);
        offer.reset(desc);
        return offer != nullptr;
    };

    if (capsFirst) {
        stream.startFrames();
        REQUIRE(WaitFor(readCaps));
        CHECK_FALSE(WaitFor(readOffer, 100ms));
    }
    remote.handleStartSession({{"sessionId", viewer.id}, {"peerId", "loopback"}});
    if (!capsFirst) {
        CHECK_FALSE(WaitFor(readOffer, 100ms));
        CHECK_FALSE(readCaps());
        stream.startFrames();
    }

    REQUIRE(WaitFor(readOffer));
    REQUIRE(offer->type == GST_WEBRTC_SDP_TYPE_OFFER);
    REQUIRE(WaitFor(readCaps));
    const char* profile = gst_structure_get_string(gst_caps_get_structure(caps.get(), 0),
                                                   "profile-level-id");
    REQUIRE(profile);
    bool video = false;
    bool control = false;
    for (guint i = 0; i < gst_sdp_message_medias_len(offer->sdp); ++i) {
        const GstSDPMedia* media = gst_sdp_message_get_media(offer->sdp, i);
        const std::string kind = gst_sdp_media_get_media(media);
        if (kind == "video") {
            video = true;
            const char* fmtp = gst_sdp_media_get_attribute_val(media, "fmtp");
            REQUIRE(fmtp);
            CHECK(std::string(fmtp).find(std::string("profile-level-id=") + profile) != std::string::npos);
        }
        if (kind == "application") control = true;
    }
    CHECK(video);
    CHECK(control);

    SetDescription(viewer.receiver, "set-remote-description", offer.get());
    Negotiate(viewer.receiver, viewer.sender, false);
    REQUIRE(WaitFor([&] { return viewer.packets >= 20; }));
}

TEST_CASE("Remote SDP validation closes rejected sessions and permits valid negotiations",
          "[core][application][remote][stream][sdp]") {
    const bool offer = GENERATE(false, true);
    const bool valid = GENERATE(false, true);
    CAPTURE(offer, valid);
    Stream stream;
    stream.start();
    Peer observer(stream, "observer");
    observer.connect();
    REQUIRE(WaitFor([&] { return observer.packets >= 20; }));
    Peer viewer(stream, "viewer");
    viewer.prepare();
    auto& remote = stream.remote;

    if (offer) {
        GstCaps* caps = gst_caps_from_string("application/x-rtp,media=video,encoding-name=H264,"
                                             "payload=96,clock-rate=90000,packetization-mode=(string)1");
        GstWebRTCRTPTransceiver* transceiver = nullptr;
        g_signal_emit_by_name(viewer.receiver, "add-transceiver",
                             GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_RECVONLY, caps, &transceiver);
        gst_caps_unref(caps);
        REQUIRE(transceiver);
        gst_object_unref(transceiver);
    } else {
        remote.createControlChannel(viewer.id);
        Negotiate(viewer.sender, viewer.receiver, true);
    }

    GstWebRTCSessionDescription* desc = CreateDescription(viewer.receiver, offer);
    if (valid) {
        SetDescription(viewer.receiver, "set-local-description", desc);
    } else {
        for (guint i = 0; i < gst_sdp_message_medias_len(desc->sdp); ++i) {
            auto* media = const_cast<GstSDPMedia*>(gst_sdp_message_get_media(desc->sdp, i));
            for (guint j = gst_sdp_media_attributes_len(media); j > 0; --j) {
                if (std::string(gst_sdp_media_get_attribute(media, j - 1)->key) == "ice-pwd") {
                    gst_sdp_media_remove_attribute(media, j - 1);
                }
            }
        }
    }
    gchar* text = gst_sdp_message_as_text(desc->sdp);
    const std::string sdp(text);
    g_free(text);
    gst_webrtc_session_description_free(desc);

    if (!valid) {
        CHECK(remote.applyRemoteDescription(viewer.id, offer ? "offer" : "answer", sdp) == Result::ERROR);
    }
    if (offer) {
        remote.handleStartSession({{"sessionId", viewer.id}, {"peerId", "loopback"}, {"offer", sdp}});
    } else {
        remote.handlePeerMessage({{"sessionId", viewer.id}, {"sdp", {{"type", "answer"}, {"sdp", sdp}}}});
    }
    CHECK(remote.sessions.contains(viewer.id) == valid);

    if (valid) {
        if (offer) {
            std::unique_ptr<GstWebRTCSessionDescription, decltype(&gst_webrtc_session_description_free)>
                answer(nullptr, gst_webrtc_session_description_free);
            REQUIRE(WaitFor([&] {
                GstWebRTCSessionDescription* local = nullptr;
                g_object_get(viewer.sender, "local-description", &local, nullptr);
                answer.reset(local);
                return answer && answer->type == GST_WEBRTC_SDP_TYPE_ANSWER;
            }));
            SetDescription(viewer.receiver, "set-remote-description", answer.get());
        }
        REQUIRE(WaitFor([&] { return viewer.packets >= 20; }));
    }
    const unsigned before = observer.packets;
    REQUIRE(WaitFor([&] { return observer.packets >= before + 20; }));
}

TEST_CASE("Remote sessions can disconnect with video blocked on SDP negotiation",
          "[core][application][remote][stream]") {
    Stream stream;
    stream.start();
    auto& remote = stream.remote;
    REQUIRE(remote.createWebRtcSession("pending", "loopback") == Result::SUCCESS);
    GstPad* pad = remote.sessions.at("pending")->webrtcSinkPad;
    REQUIRE(WaitFor([&] { return gst_pad_is_blocking(pad); }));
    remote.destroyWebRtcSession("pending");
    CHECK(remote.sessions.empty());

    Peer viewer(stream, "viewer");
    viewer.connect();
    REQUIRE(WaitFor([&] { return viewer.packets >= 20; }));
}

TEST_CASE("A stalled remote viewer does not interrupt other viewers",
          "[core][application][remote][stream][queue]") {
    const bool pending = GENERATE(false, true);
    CAPTURE(pending);
    std::atomic<unsigned> overruns = 0;
    Stream stream;
    stream.start(false);
    Peer observer(stream, "observer");
    Peer stalled(stream, "stalled");
    observer.prepare();
    stalled.prepare();

    auto& session = *stream.remote.sessions.at(stalled.id);
    g_signal_connect(session.queue, "overrun", G_CALLBACK(+[](GstElement*, gpointer data) {
        ++*static_cast<std::atomic<unsigned>*>(data);
    }), &overruns);

    stream.startFrames();
    observer.negotiate();
    PadProbe block{GST_PAD(gst_object_ref(session.webrtcSinkPad))};
    if (!pending) {
        stalled.negotiate();
        REQUIRE(WaitFor([&] { return stalled.packets >= 20; }));
        block.id = gst_pad_add_probe(block.pad,
            GstPadProbeType(GST_PAD_PROBE_TYPE_BLOCK | GST_PAD_PROBE_TYPE_BUFFER),
            +[](GstPad*, GstPadProbeInfo*, gpointer) { return GST_PAD_PROBE_OK; },
            nullptr, nullptr);
    }
    REQUIRE(WaitFor([&] { return gst_pad_is_blocking(block.pad); }));
    const unsigned receivedBefore = observer.packets;
    REQUIRE(WaitFor([&] { return overruns >= 32; }));
    REQUIRE(WaitFor([&] { return observer.packets >= receivedBefore + 20; }));
    guint queued = 0;
    g_object_get(session.queue, "current-level-buffers", &queued, nullptr);
    CHECK(queued <= 32);

    const unsigned stalledBefore = stalled.packets;
    if (pending) stalled.negotiate();
    else block.remove();
    REQUIRE(WaitFor([&] { return stalled.packets >= stalledBefore + 40; }));
}

TEST_CASE("Remote RTP delivery survives reconnects while other viewers keep receiving",
          "[core][application][remote][stream]") {
    const bool keepViewer = GENERATE(false, true);
    Stream stream;
    stream.start();
    Peer observer(stream, "observer");
    if (keepViewer) {
        observer.connect();
        REQUIRE(WaitFor([&] { return observer.packets >= 20; }));
    }
    for (unsigned i = 0; i < 8; ++i) {
        CAPTURE(keepViewer, i);
        {
            Peer viewer(stream, "viewer-" + std::to_string(i));
            viewer.connect();
            REQUIRE(WaitFor([&] { return viewer.packets >= 20; }));
        }
        if (keepViewer) {
            const unsigned before = observer.packets;
            REQUIRE(WaitFor([&] { return observer.packets >= before + 20; }));
        }
    }
}

TEST_CASE("Remote pipeline services its bus and sends RTCP sender reports",
          "[core][application][remote][stream]") {
    Stream stream;
    stream.start();
    Peer viewer(stream, "viewer");
    viewer.connect();
    REQUIRE(WaitFor([&] { return viewer.packets >= 20; }));
    REQUIRE(WaitFor([&] { return viewer.senderReports > 0; }, 10s));
    GstBus* bus = gst_element_get_bus(stream.remote.pipeline);
    for (unsigned i = 0; i < 512; ++i) {
        gst_element_post_message(stream.remote.pipeline,
            gst_message_new_application(GST_OBJECT(stream.remote.pipeline), gst_structure_new_empty("probe")));
    }
    const bool drained = WaitFor([&] { return !gst_bus_have_pending(bus); });
    gst_object_unref(bus);
    REQUIRE(drained);
}

TEST_CASE("Rejected remote submissions cannot release a later frame prematurely",
          "[core][application][remote][stream][buffer]") {
    const bool flushing = GENERATE(false, true);
    CAPTURE(flushing);
    std::atomic<bool> returned = false;
    Result submitted = Result::ERROR;
    Stream stream;
    stream.start(false);
    auto& remote = stream.remote;
    REQUIRE(remote.createWebRtcSession("viewer", "loopback") == Result::SUCCESS);
    if (flushing) {
        REQUIRE(gst_element_set_state(remote.source, GST_STATE_READY) != GST_STATE_CHANGE_FAILURE);
    } else {
        REQUIRE(gst_app_src_end_of_stream(GST_APP_SRC(remote.source)) == GST_FLOW_OK);
    }
    REQUIRE(remote.pushNewFrame(stream.frame.data()) == Result::ERROR);
    stream.stop();
    stream.start(false);
    REQUIRE(remote.createWebRtcSession("viewer", "loopback") == Result::SUCCESS);

    PadProbe block{gst_element_get_static_pad(remote.source, "src")};
    block.id = gst_pad_add_probe(block.pad,
        GstPadProbeType(GST_PAD_PROBE_TYPE_BLOCK | GST_PAD_PROBE_TYPE_BUFFER),
        +[](GstPad*, GstPadProbeInfo*, gpointer) { return GST_PAD_PROBE_OK; },
        nullptr, nullptr);
    stream.producer = std::thread([&] {
        submitted = remote.pushNewFrame(stream.frame.data());
        returned = true;
    });
    const bool held = WaitFor([&] { return gst_pad_is_blocking(block.pad); });
    const bool returnedEarly = WaitFor([&] { return returned.load(); }, 100ms);
    block.remove();
    const bool released = WaitFor([&] { return returned.load(); });
    stream.stop();
    CHECK(held);
    CHECK_FALSE(returnedEarly);
    CHECK(released);
    CHECK(submitted == Result::SUCCESS);
}

TEST_CASE("Remote pipeline can stop and restart on the same instance",
          "[core][application][remote][stream]") {
    Stream stream;
    for (unsigned i = 0; i < 3; ++i) {
        CAPTURE(i);
        stream.start();
        {
            Peer viewer(stream, "viewer");
            viewer.connect();
            REQUIRE(WaitFor([&] { return viewer.packets >= 20; }));
        }
        stream.stop();
    }
}

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
TEST_CASE("Remote NVENC shares the input CUDA context across stream restarts",
          "[core][application][remote][stream][cuda]") {
    gst_init(nullptr, nullptr);
    gst_init_static_plugins();
    GstElementFactory* factory = gst_element_factory_find("nvh264enc");
    if (!factory) {
        SKIP("NVENC is unavailable");
    }
    gst_object_unref(factory);
    if (!Backend::State<DeviceType::CUDA>()->isAvailable()) {
        SKIP("CUDA is unavailable");
    }

    Stream stream;
    auto& remote = stream.remote;
    remote.size = {320, 240};
    remote.config.codec = Instance::Remote::CodecType::H264;
    remote.inputMemoryDevice_ = DeviceType::CUDA;
    remote.encodingStrategy = Instance::Remote::Impl::EncodingStrategyType::HardwareNVENC;
    for (unsigned i = 0; i < 2; ++i) {
        CAPTURE(i);
        REQUIRE(remote.startStream() == Result::SUCCESS);
        GstPad* input = gst_element_get_static_pad(remote.encoder, "sink");
        REQUIRE(input);
        GstQuery* query = gst_query_new_context(GST_CUDA_CONTEXT_TYPE);
        const bool queried = gst_pad_query(input, query);
        GstCudaContext* encoderContext = nullptr;
        if (queried) {
            GstContext* context = nullptr;
            gst_query_parse_context(query, &context);
            if (context) {
                gst_structure_get(gst_context_get_structure(context), GST_CUDA_CONTEXT_TYPE,
                                  GST_TYPE_CUDA_CONTEXT, &encoderContext, nullptr);
            }
        }
        // Query NVENC's active context, rather than just its stored element property.
        const bool shared = encoderContext && encoderContext == remote.gstCudaContext;
        gst_clear_object(&encoderContext);
        gst_query_unref(query);
        gst_object_unref(input);
        CHECK(queried);
        CHECK(shared);
        REQUIRE(remote.stopStream() == Result::SUCCESS);
    }
}
#endif
