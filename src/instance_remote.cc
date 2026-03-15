#include "jetstream/instance_remote.hh"
#include "jetstream/viewport/capture.hh"
#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/viewport/capture/vulkan.hh"
#endif

#include <memory>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <atomic>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video-event.h>
#include <gst/gststructure.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace Jetstream {

struct Instance::Remote::Impl {
    enum class Strategy {
        None,
        Software,
        HardwareNVENC,
        HardwareV4L2,
    };

    Config config;
    Extent2D<U64> size;
    Viewport::Generic* viewport = nullptr;

    DeviceType inputMemoryDevice_ = DeviceType::None;
    Strategy encodingStrategy = Strategy::None;
    DeviceType viewportDevice = DeviceType::None;

    bool started_ = false;

    static std::string StrategyToString(const Strategy& strategy);

    // Broker state
    std::string roomId_;
    std::string consumerToken;
    std::string producerToken;
    std::string clientDomain;
    std::string signallerUrl;
    std::string inviteUrl_;
    std::vector<std::string> waitlist_;
    std::vector<ClientInfo> clients_;

    std::unique_ptr<httplib::Client> brokerClient;

    Result createBroker();
    Result destroyBroker();

    Result createRoom();

    // Stream
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* encoder = nullptr;

    bool streaming = false;

    Result createStream();
    Result startStream();
    Result stopStream();
    Result destroyStream();

    Result checkGstreamerPlugins(const std::vector<std::string>& plugins,
                                 const bool& silent = false);
    void handleInput(const std::string& kind, const nlohmann::json& j);
    static void onMessageCallback(GstWebRTCDataChannel* self, gchar* data, gpointer user_data);
    static void onChannelCallback(GstElement* self, GstWebRTCDataChannel* channel, gpointer user_data);
    static void rtcReadyCallback(GstElement* self, gchararray peer_id, GstElement* webrtcbin, gpointer udata);

    std::thread sessionMonitorThread;
    std::atomic<bool> sessionMonitorRunning{false};

    std::unique_ptr<Viewport::FrameCapture> frameCapture;
    std::thread frameSubmissionThread;
    std::atomic<bool> frameSubmissionRunning{false};

    // Frame submission
    std::mutex bufferMutex;
    std::condition_variable bufferCond;
    bool bufferProcessed = false;

    bool forceKeyframe = false;
    std::chrono::time_point<std::chrono::steady_clock> initialFrameTime;
    std::chrono::time_point<std::chrono::steady_clock> lastKeyframeTime;

    static void OnBufferReleaseCallback(gpointer user_data);
    Result pushNewFrame(const void* data);
};

Instance::Remote::Remote(Viewport::Generic* viewport) {
    impl = std::make_shared<Impl>();
    impl->viewport = viewport;
    impl->viewportDevice = viewport->device();
}

Instance::Remote::~Remote() {
    impl.reset();
}

bool Instance::Remote::supported() const {
    return impl->viewportDevice == DeviceType::Vulkan;
}

Result Instance::Remote::create(const Config& config) {
    JST_DEBUG("[REMOTE] Initializing remote streaming.");

    impl->config = config;

    if (!supported()) {
        JST_ERROR("[REMOTE] Current backend ({}) is not supported for remote streaming.", impl->viewportDevice);
        return Result::ERROR;
    }

    // Get viewport size
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (impl->viewportDevice == DeviceType::Vulkan) {
        auto* vulkanAdapter = dynamic_cast<Viewport::Adapter<DeviceType::Vulkan>*>(impl->viewport);
        if (vulkanAdapter) {
            impl->size = vulkanAdapter->getSwapchainExtent();
        }
    }
#endif

    if (impl->size.x == 0 || impl->size.y == 0) {
        JST_ERROR("[REMOTE] Failed to get viewport size.");
        return Result::ERROR;
    }

    if (impl->config.broker.empty()) {
        JST_ERROR("[REMOTE] Missing broker address.");
        return Result::ERROR;
    }

    JST_CHECK(impl->createStream());
    JST_CHECK(impl->createBroker());

    // Create frame capture
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (impl->viewportDevice == DeviceType::Vulkan) {
        impl->frameCapture = std::make_unique<Viewport::FrameCaptureVulkan>();
        JST_CHECK(impl->frameCapture->create(impl->viewport));
    }
#endif

    if (!impl->frameCapture) {
        JST_ERROR("[REMOTE] No frame capture available for device type.");
        return Result::ERROR;
    }

    // Start frame submission thread
    impl->frameSubmissionRunning = true;
    impl->frameSubmissionThread = std::thread([this]() {
        while (impl->frameSubmissionRunning) {
            Tensor tensor;
            if (impl->frameCapture->getFrameData(tensor) == Result::SUCCESS) {
                impl->pushNewFrame(tensor.data());
                impl->frameCapture->releaseFrame();
            }
        }
    });

    impl->sessionMonitorRunning = true;
    impl->sessionMonitorThread = std::thread([this]() {
        while (impl->sessionMonitorRunning) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (!impl->sessionMonitorRunning) break;
            updateWaitlist();
            updateSessions();
        }
    });

    impl->started_ = true;
    JST_INFO("[REMOTE] Remote streaming started.");
    return Result::SUCCESS;
}

Result Instance::Remote::destroy() {
    JST_DEBUG("[REMOTE] Destroying remote streaming.");

    impl->sessionMonitorRunning = false;
    if (impl->sessionMonitorThread.joinable()) {
        impl->sessionMonitorThread.join();
    }

    impl->frameSubmissionRunning = false;
    if (impl->frameCapture) {
        impl->frameCapture->stop();
    }
    if (impl->frameSubmissionThread.joinable()) {
        impl->frameSubmissionThread.join();
    }

    if (impl->frameCapture) {
        impl->frameCapture->destroy();
        impl->frameCapture.reset();
    }

    JST_CHECK(impl->destroyStream());
    JST_CHECK(impl->destroyBroker());

    impl->started_ = false;
    JST_INFO("[REMOTE] Remote streaming stopped.");
    return Result::SUCCESS;
}

bool Instance::Remote::started() const {
    return impl->started_;
}

Result Instance::Remote::captureFrame() {
    if (impl->frameCapture) {
        impl->frameCapture->captureFrame();
    }
    return Result::SUCCESS;
}

const std::string& Instance::Remote::roomId() const {
    return impl->roomId_;
}

const std::string& Instance::Remote::accessToken() const {
    return impl->consumerToken;
}

const std::string& Instance::Remote::inviteUrl() const {
    return impl->inviteUrl_;
}

const std::vector<Instance::Remote::ClientInfo>& Instance::Remote::clients() const {
    return impl->clients_;
}

const std::vector<std::string>& Instance::Remote::waitlist() const {
    return impl->waitlist_;
}

Result Instance::Remote::updateWaitlist() {
    if (!impl->started_) {
        return Result::SUCCESS;
    }

    auto res = impl->brokerClient->Get("/api/v1/remote/room/waitlist");
    if (!res || res->status != 200) {
        JST_ERROR("[REMOTE] Failed to pull waitlist: [{}] /api/v1/remote/room/waitlist.", res ? res->status : 0);
        return Result::ERROR;
    }

    try {
        auto j = nlohmann::json::parse(res->body);

        if (!j.contains("sessions")) {
            JST_ERROR("[REMOTE] Missing field 'sessions': {}", res->body);
            return Result::ERROR;
        }

        impl->waitlist_ = j["sessions"].get<std::vector<std::string>>();
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::updateSessions() {
    if (!impl->started_) {
        return Result::SUCCESS;
    }

    auto res = impl->brokerClient->Get("/api/v1/remote/room/active");
    if (!res || res->status != 200) {
        JST_ERROR("[REMOTE] Failed to pull sessions: [{}] /api/v1/remote/room/active.", res ? res->status : 0);
        return Result::ERROR;
    }

    try {
        auto j = nlohmann::json::parse(res->body);

        if (!j.contains("sessions")) {
            JST_ERROR("[REMOTE] Missing field 'sessions': {}", res->body);
            return Result::ERROR;
        }

        auto sessions = j["sessions"].get<std::vector<std::string>>();
        impl->clients_.clear();
        for (const auto& s : sessions) {
            impl->clients_.push_back({s});
        }
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::approveClient(const std::string& code) {
    if (!impl->started_) {
        JST_ERROR("[REMOTE] Can't approve client when session is not started.");
        return Result::ERROR;
    }

    JST_CHECK(updateWaitlist());

    auto to_lower = [](std::string_view s) {
        std::string out(s);
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return out;
    };

    for (auto& sessionId : impl->waitlist_) {
        JST_DEBUG("[REMOTE] Candidate session: {}", sessionId);

        if (sessionId.ends_with(to_lower(code))) {
            JST_INFO("[REMOTE] Client authorization code '{}' approved.", code);

            auto json = nlohmann::json{{"sessionId", sessionId}};
            auto res = impl->brokerClient->Post("/api/v1/remote/room/approval", json.dump(), "application/json");
            if (!res || res->status != 200) {
                JST_ERROR("[REMOTE] Failed to post approval: [{}] /api/v1/remote/room/approval.", res ? res->status : 0);
                return Result::ERROR;
            }

            JST_CHECK(updateSessions());
            return Result::SUCCESS;
        }
    }

    JST_ERROR("[REMOTE] Client authorization code '{}' not found.", code);
    return Result::ERROR;
}

//
// Broker
//

Result Instance::Remote::Impl::createBroker() {
    JST_INFO("[REMOTE] Connecting to broker at '{}'.", config.broker);
    brokerClient = std::make_unique<httplib::Client>(config.broker);

    {
        auto res = brokerClient->Get("/api/v1");
        if (!res || res->status != 200) {
            JST_ERROR("[REMOTE] Failed to connect to server.");
            return Result::ERROR;
        }
        JST_DEBUG("[REMOTE] Connected to server.");
    }

    JST_CHECK(createRoom());

    JST_INFO("[REMOTE] Signaller URL: '{}'.", signallerUrl);

    brokerClient->set_bearer_token_auth(producerToken);

    inviteUrl_ = jst::fmt::format("{}#{}", clientDomain, consumerToken);

    JST_CHECK(startStream());

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::destroyBroker() {
    JST_DEBUG("[REMOTE] Closing broker connection.");
    JST_CHECK(stopStream());
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::createRoom() {
    auto params = httplib::Params{};
    auto res = brokerClient->Post("/api/v1/remote/room", params);
    if (!res || res->status != 201) {
        JST_ERROR("[REMOTE] Failed to create room: [{}] /api/v1/remote/room.", res ? res->status : 0);
        return Result::ERROR;
    }

    try {
        auto j = nlohmann::json::parse(res->body);

        if (!j.contains("roomId")) {
            JST_ERROR("[REMOTE] Missing field 'roomId': {}", res->body);
            return Result::ERROR;
        }

        if (!j.contains("producerToken")) {
            JST_ERROR("[REMOTE] Missing field 'producerToken': {}", res->body);
            return Result::ERROR;
        }

        if (!j.contains("consumerToken")) {
            JST_ERROR("[REMOTE] Missing field 'consumerToken': {}", res->body);
            return Result::ERROR;
        }

        if (!j.contains("signallerUrl")) {
            JST_ERROR("[REMOTE] Missing field 'signallerUrl': {}", res->body);
            return Result::ERROR;
        }

        if (!j.contains("clientDomain")) {
            JST_ERROR("[REMOTE] Missing field 'clientDomain': {}", res->body);
            return Result::ERROR;
        }

        roomId_ = j["roomId"].get<std::string>();
        producerToken = j["producerToken"].get<std::string>();
        consumerToken = j["consumerToken"].get<std::string>();
        signallerUrl = j["signallerUrl"].get<std::string>();
        clientDomain = j["clientDomain"].get<std::string>();
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    JST_DEBUG("[REMOTE] New room created.");
    return Result::SUCCESS;
}

//
// Stream
//

Result Instance::Remote::Impl::createStream() {
    JST_DEBUG("[REMOTE] Creating stream.");

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    std::vector<std::string> plugins = {
        "app",
        "rawparse",
        "coreelements",
    };

    plugins.push_back("rswebrtc");

    JST_CHECK(checkGstreamerPlugins(plugins));

    std::vector<std::tuple<DeviceType, Strategy, std::vector<std::string>>> combinations;

    if (config.codec == Viewport::VideoCodec::H264) {
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (viewportDevice == DeviceType::Vulkan && Backend::State<DeviceType::CUDA>()->isAvailable()) {
            combinations.push_back({DeviceType::CUDA, Strategy::HardwareNVENC, {"nvcodec"}});
            GST_DEBUG("[REMOTE] Checking for NVENC strategy support for h264.");
        }
#endif

        if (checkGstreamerPlugins({"video4linux2"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("v4l2h264enc");
            if (factory) {
                combinations.push_back({DeviceType::CPU, Strategy::HardwareV4L2, {"video4linux2"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[REMOTE] Checking for V4L2 strategy support for h264.");
            }
        }

        combinations.push_back({DeviceType::CPU, Strategy::Software, {"x264"}});
    }

    if (config.codec == Viewport::VideoCodec::VP8) {
        combinations.push_back({DeviceType::CPU, Strategy::Software, {"vpx"}});
    }

    if (config.codec == Viewport::VideoCodec::VP9) {
        combinations.push_back({DeviceType::CPU, Strategy::Software, {"vpx"}});
    }

    if (config.codec == Viewport::VideoCodec::AV1) {
        combinations.push_back({DeviceType::CPU, Strategy::Software, {"rav1e"}});
    }

    for (const auto& [device, strategy, pluginList] : combinations) {
        if ((strategy != Strategy::Software) && !config.hardwareAcceleration) {
            continue;
        }
        if (checkGstreamerPlugins(pluginList, true) == Result::SUCCESS) {
            inputMemoryDevice_ = device;
            encodingStrategy = strategy;

            JST_INFO("[REMOTE] Using {} encoding with {} memory.", StrategyToString(strategy),
                                                                   GetDevicePrettyName(device));

            return Result::SUCCESS;
        }
        JST_DEBUG("[REMOTE] Failed to find plugins: {}", pluginList);
    }

    JST_ERROR("[REMOTE] No encoding combination is available.");
    JST_ERROR("[REMOTE] This is typically caused by missing plugins.");
    return Result::ERROR;
}

Result Instance::Remote::Impl::destroyStream() {
    JST_DEBUG("[REMOTE] Destroying stream.");

    encodingStrategy = Strategy::None;
    inputMemoryDevice_ = DeviceType::None;

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::checkGstreamerPlugins(const std::vector<std::string>& plugins,
                                                     const bool& silent) {
    for (const auto& plugin : plugins) {
        if (!gst_registry_find_plugin(gst_registry_get(), plugin.c_str())) {
            if (!silent) {
                JST_ERROR("[REMOTE] Gstreamer plugin '{}' is not available.", plugin);
            }
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

void Instance::Remote::Impl::handleInput(const std::string& kind, const nlohmann::json& j) {
    auto mapMouseButton = [](int domButton) -> int {
        switch (domButton) {
            case 0: return 0;
            case 1: return 2;
            case 2: return 1;
            default: return domButton;
        }
    };

    auto denorm = [](double n, int extent) -> float {
        if (extent <= 0) return 0.0f;
        if (n >= 0.0 && n <= 1.0) {
            return static_cast<float>(std::round(n * std::max(0, extent - 1)));
        }
        return static_cast<float>(std::round(n));
    };

    auto firstCodepoint = [](const std::string& s) -> unsigned int {
        if (s.empty()) return 0u;
        if (s.size() > 1) return 0u;

        const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
        if (p[0] < 0x80) return static_cast<unsigned int>(p[0]);
        if ((p[0] & 0xE0) == 0xC0 && s.size() >= 2) {
            return ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        }
        if ((p[0] & 0xF0) == 0xE0 && s.size() >= 3) {
            return ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        }
        if ((p[0] & 0xF8) == 0xF0 && s.size() >= 4) {
            return ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        }
        return 0u;
    };

    auto mapKey = [](const std::string& code, const std::string& key) -> ImGuiKey {
        if (code.size() == 4 && code.rfind("Key", 0) == 0) {
            char c = static_cast<char>(std::toupper(static_cast<unsigned char>(code[3])));
            if (c >= 'A' && c <= 'Z') return static_cast<ImGuiKey>(ImGuiKey_A + (c - 'A'));
        }
        if (code.rfind("Digit", 0) == 0 && code.size() == 6 && std::isdigit(static_cast<unsigned char>(code[5]))) {
            char d = code[5];
            return static_cast<ImGuiKey>(ImGuiKey_0 + (d - '0'));
        }
        if (!code.empty() && code[0] == 'F' && code.size() <= 3 && std::isdigit(static_cast<unsigned char>(code[1]))) {
            int f = std::clamp(std::stoi(code.substr(1)), 1, 24);
            return static_cast<ImGuiKey>(ImGuiKey_F1 + (f - 1));
        }
        if (code == "ArrowLeft")                       return ImGuiKey_LeftArrow;
        if (code == "ArrowRight")                      return ImGuiKey_RightArrow;
        if (code == "ArrowUp")                         return ImGuiKey_UpArrow;
        if (code == "ArrowDown")                       return ImGuiKey_DownArrow;
        if (code == "Enter" || key == "Enter")         return ImGuiKey_Enter;
        if (code == "Escape" || key == "Escape")       return ImGuiKey_Escape;
        if (code == "Backspace" || key == "Backspace") return ImGuiKey_Backspace;
        if (code == "Tab" || key == "Tab")             return ImGuiKey_Tab;
        if (code == "Space" || key == " ")             return ImGuiKey_Space;
        if (code == "Delete")                          return ImGuiKey_Delete;
        if (code == "Insert")                          return ImGuiKey_Insert;
        if (code == "Home")                            return ImGuiKey_Home;
        if (code == "End")                             return ImGuiKey_End;
        if (code == "PageUp")                          return ImGuiKey_PageUp;
        if (code == "PageDown")                        return ImGuiKey_PageDown;
        if (code == "Minus")                           return ImGuiKey_Minus;
        if (code == "Equal")                           return ImGuiKey_Equal;
        if (code == "BracketLeft")                     return ImGuiKey_LeftBracket;
        if (code == "BracketRight")                    return ImGuiKey_RightBracket;
        if (code == "Backslash")                       return ImGuiKey_Backslash;
        if (code == "Semicolon")                       return ImGuiKey_Semicolon;
        if (code == "Quote")                           return ImGuiKey_Apostrophe;
        if (code == "Comma")                           return ImGuiKey_Comma;
        if (code == "Period")                          return ImGuiKey_Period;
        if (code == "Slash")                           return ImGuiKey_Slash;
        if (code == "Backquote" ||
            code == "Backtick"  ||
            code == "QuoteLeft")                       return ImGuiKey_GraveAccent;
        return ImGuiKey_None;
    };

    if (kind == "mouse") {
        const std::string act = j.value("act", "");
        const double nx = j.value("x", 0.0);
        const double ny = j.value("y", 0.0);

        const int w = size.x;
        const int h = size.y;

        ImGuiIO& io = ImGui::GetIO();
        const float px = denorm(nx, w) / io.DisplayFramebufferScale.x;
        const float py = denorm(ny, h) / io.DisplayFramebufferScale.y;

        if (act == "move") {
            JST_TRACE("[REMOTE] Mouse: move (x='{}', y='{}', nx='{}', ny='{}')", px, py, nx, ny);
            io.AddMousePosEvent(px, py);
            return;
        }

        if (act == "down" || act == "up" || act == "click" || act == "dblclick") {
            const int domButton = j.value("button", 0);
            const int b = mapMouseButton(domButton);

            io.AddMousePosEvent(px, py);

            if (act == "down") {
                JST_TRACE("[REMOTE] Mouse: down (b='{}', x='{}', y='{}')", b, px, py);
                io.AddMouseButtonEvent(b, true);
                return;
            }
            if (act == "up") {
                JST_TRACE("[REMOTE] Mouse: up b={}, x={}, y={}", b, px, py);
                io.AddMouseButtonEvent(b, false);
                return;
            }
            if (act == "click" || act == "dblclick") {
                JST_TRACE("[REMOTE] Mouse: ignore (b='{}', x='{}', y='{}')", b, px, py);
                return;
            }
        }

        JST_TRACE("[REMOTE] Mouse: unknown (act='{}')", act);
        return;
    }

    if (kind == "wheel") {
        const int   deltaMode = j.value("deltaMode", 0);
        double      dx = j.value("deltaX", 0.0);
        double      dy = j.value("deltaY", 0.0);

        if (deltaMode == 0) { dx /= 100.0; dy /= 100.0; }
        else if (deltaMode == 2) { dx *= 3.0; dy *= 3.0; }

        dy = -dy;

        ImGuiIO& io = ImGui::GetIO();
        io.AddMouseWheelEvent(static_cast<float>(dx), static_cast<float>(dy));
        JST_TRACE("[REMOTE] Wheel: (dx='{}', dy='{}', mode='{}')", dx, dy, deltaMode);
        return;
    }

    const bool alt   = j.value("altKey",   false);
    const bool ctrl  = j.value("ctrlKey",  false);
    const bool shift = j.value("shiftKey", false);
    const bool meta  = j.value("metaKey",  false);

    if (kind == "keyboard") {
        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiKey_LeftAlt, alt);
        io.AddKeyEvent(ImGuiKey_LeftCtrl, ctrl);
        io.AddKeyEvent(ImGuiKey_LeftShift, shift);
        io.AddKeyEvent(ImGuiKey_LeftSuper, meta);

        const std::string action = j.value("action", "");
        const std::string code = j.value("code", "");
        const std::string key = j.value("key", "");
        const bool pressed = (action == "down");

        const ImGuiKey k = mapKey(code, key);
        if (k != ImGuiKey_None) {
            ImGui::GetIO().AddKeyEvent(k, pressed);
        }
        JST_TRACE("[REMOTE] Keyboard: event (pressed='{}', key='{}', code='{}', ImGuiKey='{}')", pressed ? "down" : "up", code, key, (int)k);

        if (pressed && !ctrl && !alt && !meta) {
            const unsigned int cp = firstCodepoint(key);
            if (cp >= 0x20 && cp != 0x7F) {
                ImGui::GetIO().AddInputCharacter(cp);
                JST_TRACE("[REMOTE] Keyboard: char (key='{}', code='U+{:04X}')", key, cp);
            }
        }
        return;
    }

    JST_TRACE("[REMOTE] Unknown control (kind='{}').", kind);
}

void Instance::Remote::Impl::onMessageCallback(GstWebRTCDataChannel* self, gchar* data, gpointer user_data) {
    (void)self;

    JST_TRACE("[REMOTE] Received string: {}", data);

    nlohmann::json j;

    try {
        j = nlohmann::json::parse(data);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Bad control JSON: {} (payload='{}')", e.what(), data);
        return;
    }

    const std::string kind = j.value("kind", "");
    if (kind.empty()) {
        JST_WARN("[REMOTE] Control msg without 'kind': '{}'", data);
        return;
    }

    auto* that = reinterpret_cast<Instance::Remote::Impl*>(user_data);
    that->handleInput(kind, j);
}

void Instance::Remote::Impl::onChannelCallback(GstElement* self, GstWebRTCDataChannel* channel, gpointer user_data) {
    (void)self;
    (void)user_data;

    gchar *label = NULL; gint id = -1; gboolean negotiated = FALSE;
    g_object_get(channel, "label", &label, "id", &id, "negotiated", &negotiated, NULL);
    JST_INFO("[REMOTE] Dropping data channel opened from client (label='{}', id='{}', negotiated='{}')", label, id, negotiated);

    g_signal_emit_by_name(channel, "close");
    g_free(label);
}

void Instance::Remote::Impl::rtcReadyCallback(GstElement* self, gchararray peer_id, GstElement* webrtcbin, gpointer udata) {
    (void)self;

    JST_INFO("[REMOTE] New consumer ({}) connected.", peer_id);

    g_signal_connect(G_OBJECT(webrtcbin), "on-data-channel", G_CALLBACK(onChannelCallback), udata);

    GstWebRTCDataChannel* channel = NULL;
    g_signal_emit_by_name(G_OBJECT(webrtcbin), "create-data-channel", "control", nullptr, &channel);
    if (!channel) {
        JST_ERROR("[REMOTE] Failed to create data channel.");
        return;
    }
    g_signal_connect(G_OBJECT(channel), "on-message-string", G_CALLBACK(onMessageCallback), udata);
    g_object_unref(channel);
}

Result Instance::Remote::Impl::startStream() {
    JST_DEBUG("[REMOTE] Starting stream.");

    pipeline = gst_pipeline_new("remote-src-pipeline");

    if (!pipeline) {
        JST_ERROR("[REMOTE] Failed to create gstreamer pipeline.");
        return Result::ERROR;
    }

    std::map<std::string, GstElement*> elements;
    std::vector<std::string> elementOrder;

    // 01. Setup Source element.

    elements["source"] = source = gst_element_factory_make("appsrc", "source");
    elementOrder.push_back("source");

    g_object_set(elements["source"], "block", true, nullptr);
    g_object_set(elements["source"], "format", 3, nullptr);
    g_object_set(elements["source"], "leaky-type", 2, nullptr);
    g_object_set(elements["source"], "is-live", true, nullptr);
    g_object_set(elements["source"], "max-bytes", 2*size.x*size.y*4, nullptr);

    // 02. Setup Caps element.

    elements["caps"] = gst_element_factory_make("capsfilter", "caps");
    elementOrder.push_back("caps");

    GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "BGRA",
                                        "width", G_TYPE_INT, static_cast<int>(size.x),
                                        "height", G_TYPE_INT, static_cast<int>(size.y),
                                        "framerate", GST_TYPE_FRACTION, config.framerate, 1,
                                        "interlace-mode", G_TYPE_STRING, "progressive",
                                        "colorimetry", G_TYPE_STRING, "bt709",
                                        nullptr);

    switch (inputMemoryDevice_) {
        case DeviceType::CPU:
            break;
        case DeviceType::CUDA: {
            GstCapsFeatures* features = gst_caps_features_new("memory:CUDAMemory", nullptr);
            gst_caps_set_features(caps, 0, features);
            break;
        }
        default:
            JST_ERROR("[REMOTE] Unsupported input memory device '{}'.", inputMemoryDevice_);
            gst_caps_unref(caps);
            return Result::ERROR;
    }

    g_object_set(elements["caps"], "caps", caps, nullptr);
    gst_caps_unref(caps);

    // 03. Setup strategy-specific elements.

    if (encodingStrategy == Strategy::None) {
        JST_ERROR("[REMOTE] No encoding strategy selected.");
        return Result::ERROR;
    }

    if (encodingStrategy == Strategy::Software) {
        elements["rawparser"] = gst_element_factory_make("rawvideoparse", "rawparser");
        elementOrder.push_back("rawparser");

        g_object_set(elements["rawparser"], "use-sink-caps", 0, nullptr);
        g_object_set(elements["rawparser"], "format", 12, nullptr);
        g_object_set(elements["rawparser"], "width", static_cast<int>(size.x), nullptr);
        g_object_set(elements["rawparser"], "height", static_cast<int>(size.y), nullptr);

        elements["convert"] = gst_element_factory_make("videoconvert", "convert");
        elementOrder.push_back("convert");

        switch(config.codec) {
            case Viewport::VideoCodec::H264: {
                elements["encoder"] = encoder = gst_element_factory_make("x264enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "speed-preset", 1, nullptr);
                g_object_set(elements["encoder"], "tune", 4, nullptr);
                g_object_set(elements["encoder"], "bitrate", 25*1024*1024, nullptr);

                elements["hwcaps"] = gst_element_factory_make("capsfilter", "hwcaps");
                elementOrder.push_back("hwcaps");

                GstCaps* hwcaps = gst_caps_new_simple("video/x-h264",
                                                      "profile", G_TYPE_STRING, "high",
                                                      nullptr);
                g_object_set(elements["hwcaps"], "caps", hwcaps, nullptr);
                gst_caps_unref(hwcaps);

                elements["parser"] = gst_element_factory_make("h264parse", "parser");
                elementOrder.push_back("parser");

                g_object_set(elements["parser"], "config-interval", 1, nullptr);
                break;
            }
            case Viewport::VideoCodec::VP8:
                elements["encoder"] = encoder = gst_element_factory_make("vp8enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "target-bitrate", 25*1024*1024, nullptr);
                break;
            case Viewport::VideoCodec::VP9:
                elements["encoder"] = encoder = gst_element_factory_make("vp9enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "target-bitrate", 25*1024*1024, nullptr);
                break;
            case Viewport::VideoCodec::AV1:
                elements["encoder"] = encoder = gst_element_factory_make("rav1enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "low-latency", true, nullptr);
                g_object_set(elements["encoder"], "speed-preset", 10, nullptr);
                g_object_set(elements["encoder"], "bitrate", 25*1024*1024, nullptr);
                break;
            default:
                JST_ERROR("[REMOTE] Unsupported codec for software encoding.");
                return Result::ERROR;
        }
    }

    if (encodingStrategy == Strategy::HardwareNVENC) {
        switch(config.codec) {
            case Viewport::VideoCodec::H264:
                elements["encoder"] = encoder = gst_element_factory_make("nvh264enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "zerolatency", true, nullptr);
                g_object_set(elements["encoder"], "preset", 5, nullptr);

                elements["hwcaps"] = gst_element_factory_make("capsfilter", "hwcaps");
                elementOrder.push_back("hwcaps");

                elements["parser"] = gst_element_factory_make("h264parse", "parser");
                elementOrder.push_back("parser");

                g_object_set(elements["parser"], "config-interval", 1, nullptr);
                break;
            default:
                JST_ERROR("[REMOTE] Unsupported codec for hardware encoding.");
                return Result::ERROR;
        }
    }

    if (encodingStrategy == Strategy::HardwareV4L2) {
        elements["rawparser"] = gst_element_factory_make("rawvideoparse", "rawparser");
        elementOrder.push_back("rawparser");

        g_object_set(elements["rawparser"], "use-sink-caps", 0, nullptr);
        g_object_set(elements["rawparser"], "format", 12, nullptr);
        g_object_set(elements["rawparser"], "width", static_cast<int>(size.x), nullptr);
        g_object_set(elements["rawparser"], "height", static_cast<int>(size.y), nullptr);

        elements["convert"] = gst_element_factory_make("videoconvert", "convert");
        elementOrder.push_back("convert");

        switch(config.codec) {
            case Viewport::VideoCodec::H264: {
                elements["encoder"] = encoder = gst_element_factory_make("v4l2h264enc", "encoder");
                elementOrder.push_back("encoder");

                elements["hwcaps"] = gst_element_factory_make("capsfilter", "hwcaps");
                elementOrder.push_back("hwcaps");

                GstCaps* hwcaps = gst_caps_new_simple("video/x-h264",
                                                      "profile", G_TYPE_STRING, "high",
                                                      nullptr);
                g_object_set(elements["hwcaps"], "caps", hwcaps, nullptr);
                gst_caps_unref(hwcaps);

                elements["parser"] = gst_element_factory_make("h264parse", "parser");
                elementOrder.push_back("parser");

                g_object_set(elements["parser"], "config-interval", 1, nullptr);
                break;
            }
            default:
                JST_ERROR("[REMOTE] Unsupported codec for hardware encoding.");
                return Result::ERROR;
        }
    }

    // 04. Setup WebRTC element.

    elements["webrtc"] = gst_element_factory_make("webrtcsink", "webrtc");
    elementOrder.push_back("webrtc");

    GstStructure *s = gst_structure_new_empty("meta");
    gst_structure_set(s, "token", G_TYPE_STRING, producerToken.c_str(), nullptr);
    g_object_set(elements["webrtc"], "meta", s, NULL);
    gst_structure_free(s);

    GObject* signaller;
    g_object_get(elements["webrtc"], "signaller", &signaller, nullptr);
    if (signaller) {
        JST_DEBUG("[REMOTE] Setting signaller URI to {}", signallerUrl);
        g_object_set(signaller, "uri", signallerUrl.c_str(), nullptr);
        g_signal_connect(G_OBJECT(signaller), "webrtcbin-ready", G_CALLBACK(rtcReadyCallback), this);
        g_object_unref(signaller);
    } else {
        JST_ERROR("[REMOTE] Failed to get signaller object from WebRTC element.");
        return Result::ERROR;
    }

    // Add elements to the pipeline.

    for (const auto& [name, element] : elements) {
        if (!gst_bin_add(GST_BIN(pipeline), element)) {
            JST_ERROR("[REMOTE] Failed to add gstreamer element '{}' to pipeline.", name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }
    }

    // Link elements in the pipeline.

    std::string lastElement = "source";
    for (const auto& name : elementOrder) {
        if (name == "source") {
            continue;
        }

        if (!gst_element_link(elements[lastElement], elements[name])) {
            JST_ERROR("[REMOTE] Failed to link gstreamer element '{}' -> '{}'.", lastElement, name);
            gst_object_unref(pipeline);
            return Result::ERROR;
        }

        lastElement = name;
    }

    // Set pipeline state to playing.

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        JST_ERROR("[REMOTE] Failed to start gstreamer pipeline.");
        gst_object_unref(pipeline);
        return Result::ERROR;
    }

    initialFrameTime = std::chrono::steady_clock::now();
    forceKeyframe = true;
    streaming = true;

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::stopStream() {
    JST_DEBUG("[REMOTE] Stopping stream.");

    if (streaming) {
        streaming = false;

        gst_element_send_event(pipeline, gst_event_new_eos());

        GstBus* bus = gst_element_get_bus(pipeline);
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS);
        if (msg) {
            gst_message_unref(msg);
        }
        gst_object_unref(bus);

        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }

    return Result::SUCCESS;
}

//
// Frame Submission
//

void Instance::Remote::Impl::OnBufferReleaseCallback(gpointer user_data) {
    auto* that = reinterpret_cast<Instance::Remote::Impl*>(user_data);
    std::unique_lock<std::mutex> lock(that->bufferMutex);
    that->bufferProcessed = true;
    that->bufferCond.notify_one();
}

Result Instance::Remote::Impl::pushNewFrame(const void* data) {
    if (!streaming) {
        return Result::SUCCESS;
    }

    GstBuffer* buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                    const_cast<void*>(data),
                                                    size.x * size.y * 4,
                                                    0,
                                                    size.x * size.y * 4,
                                                    this,
                                                    &OnBufferReleaseCallback);

    const auto currentFrameTime = std::chrono::steady_clock::now();
    const auto elapsedSinceLastFrame = std::chrono::duration_cast<std::chrono::nanoseconds>(currentFrameTime -
                                                                                            initialFrameTime);
    const auto elapsedSinceLastKeyframe = std::chrono::duration_cast<std::chrono::seconds>(currentFrameTime -
                                                                                            lastKeyframeTime);

    GST_BUFFER_PTS(buffer) = static_cast<U64>(elapsedSinceLastFrame.count());
    GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;

    if ((elapsedSinceLastKeyframe.count() > 1) || forceKeyframe) {
        GstEvent* force_key_unit_event = gst_video_event_new_downstream_force_key_unit(
            GST_CLOCK_TIME_NONE,
            GST_CLOCK_TIME_NONE,
            GST_CLOCK_TIME_NONE,
            TRUE,
            0
        );

        gst_element_send_event(encoder, force_key_unit_event);

        lastKeyframeTime = currentFrameTime;
        forceKeyframe = false;
    }

    if (gst_app_src_push_buffer(GST_APP_SRC(source), buffer) != GST_FLOW_OK) {
        JST_ERROR("[REMOTE] Failed to push buffer to gstreamer pipeline.");
        return Result::ERROR;
    }

    {
        std::unique_lock<std::mutex> lock(bufferMutex);
        bufferCond.wait(lock, [this]{ return bufferProcessed; });
        bufferProcessed = false;
    }

    return Result::SUCCESS;
}

//
// Helpers
//

std::string Instance::Remote::Impl::StrategyToString(const Strategy& strategy) {
    switch (strategy) {
        case Strategy::None:
            return "None";
        case Strategy::Software:
            return "Software";
        case Strategy::HardwareNVENC:
            return "Hardware NVIDIA (NVENC)";
        case Strategy::HardwareV4L2:
            return "Hardware Linux (V4L2)";
        default:
            return "Unknown";
    }
}

}  // namespace Jetstream
