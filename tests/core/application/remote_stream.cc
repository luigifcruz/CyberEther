#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../../../src/instance_remote_impl.hh"

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/backend/base.hh"
#endif

#include <gst/app/gstappsink.h>
#include <gst/rtp/gstrtcpbuffer.h>

using namespace Jetstream;
using namespace std::chrono_literals;

extern "C" void gst_init_static_plugins();

namespace {

template<typename Predicate>
bool WaitFor(Predicate predicate, std::chrono::seconds timeout = 3s) {
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

    void start() {
        gst_init(nullptr, nullptr);
        gst_init_static_plugins();
        remote.size = {320, 240};
        remote.config.codec = Instance::Remote::CodecType::H264;
        remote.inputMemoryDevice_ = DeviceType::CPU;
        remote.encodingStrategy = Instance::Remote::Impl::EncodingStrategyType::Software;
        REQUIRE(remote.startStream() == Result::SUCCESS);
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
        remote.stopStream();
        if (producer.joinable()) producer.join();
        remote.destroyStream();
    }

    ~Stream() { stop(); }
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

void Negotiate(GstElement* from, GstElement* to, bool offer) {
    GstPromise* promise = gst_promise_new();
    g_signal_emit_by_name(from, offer ? "create-offer" : "create-answer", nullptr, promise);
    REQUIRE(gst_promise_wait(promise) == GST_PROMISE_RESULT_REPLIED);
    GstWebRTCSessionDescription* desc = nullptr;
    gst_structure_get(gst_promise_get_reply(promise), offer ? "offer" : "answer",
                      GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &desc, nullptr);
    gst_promise_unref(promise);
    REQUIRE(desc);
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

    void connect() {
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

}  // namespace

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
