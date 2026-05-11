#include "jetstream/instance_remote.hh"
#include "jetstream/viewport/capture.hh"
#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/viewport/capture/vulkan.hh"
#endif

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>
#include <gst/app/gstappsrc.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video-event.h>
#include <gst/gststructure.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace Jetstream {

struct Instance::Remote::Impl {
    ~Impl() = default;

    bool supported() const;
    Result create(const Instance::Remote::Config& config);
    Result destroy();
    Result captureFrame();
    Result updateWaitlist();
    Result updateSessions();
    Result approveClient(const std::string& code);

    Config config;
    Viewport::Generic* viewport = nullptr;
    DeviceType viewportDevice = DeviceType::None;
    bool started_ = false;
    std::string roomId_;
    std::string consumerToken;
    std::string inviteUrl_;
    std::vector<std::string> waitlist_;
    std::vector<ClientInfo> clients_;

    enum class EncodingStrategyType {
        None,
        Software,
        HardwareNVENC,
        HardwareV4L2,
        HardwareVideoToolbox,
    };

    Extent2D<U64> size;

    DeviceType inputMemoryDevice_ = DeviceType::None;
    EncodingStrategyType encodingStrategy = EncodingStrategyType::None;

    // Broker state
    std::string producerToken;
    std::string clientDomain;
    std::string signallerUrl;

    std::unique_ptr<httplib::Client> brokerClient;

    Result createBroker();
    Result destroyBroker();

    Result createRoom();

    // Stream
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* encoder = nullptr;
    GstElement* tee = nullptr;

    std::unique_ptr<httplib::ws::WebSocketClient> signallerClient;
    std::thread signallerThread;
    std::atomic<bool> signallerRunning = false;
    std::mutex signallerMutex;
    std::string producerPeerId;

    struct WebRtcSession {
        std::string sessionId;
        std::string peerId;
        GstElement* queue = nullptr;
        GstElement* payloader = nullptr;
        GstElement* rtpCaps = nullptr;
        GstElement* webrtc = nullptr;
        GstPad* teeSrcPad = nullptr;
        GstPad* webrtcSinkPad = nullptr;
        gulong iceHandler = 0;
        gulong channelHandler = 0;
    };

    struct WebRtcSignalContext {
        Impl* impl = nullptr;
        std::string sessionId;
    };

    std::mutex sessionsMutex;
    std::unordered_map<std::string, std::unique_ptr<WebRtcSession>> sessions;

    std::atomic<bool> streaming = false;
    std::mutex streamMutex;

    Result createStream();
    Result startStream();
    Result stopStream();
    Result destroyStream();
    Result createWebRtcSession(const std::string& sessionId, const std::string& peerId);
    void destroyWebRtcSession(const std::string& sessionId);
    void destroyAllWebRtcSessions();
    GstElement* refSessionWebrtc(const std::string& sessionId);
    GstElement* createPayloader();
    GstElement* createRtpCapsFilter();
    Result startSignaller();
    Result stopSignaller();
    void signallerLoop();
    void handleSignallerMessage(const std::string& payload);
    void handleStartSession(const nlohmann::json& j);
    void handlePeerMessage(const nlohmann::json& j);
    void handleEndSession(const nlohmann::json& j);
    bool sendSignallerMessage(const nlohmann::json& j);
    void sendSessionDescription(const std::string& sessionId, const GstWebRTCSessionDescription* desc);
    void sendIceCandidate(const std::string& sessionId, guint mlineIndex, const gchar* candidate);
    Result applyRemoteDescription(const std::string& sessionId,
                                  const std::string& type,
                                  const std::string& sdp);

    Result checkGstreamerPlugins(const std::vector<std::string>& plugins,
                                 const bool& silent = false);
    static const char* GetEncodingStrategyPrettyName(const EncodingStrategyType& strategy);
    void handleInput(const std::string& kind, const nlohmann::json& j);
    void createControlChannel(const std::string& sessionId);
    static void onMessageCallback(GstWebRTCDataChannel* self, gchar* data, gpointer user_data);
    static void onChannelCallback(GstElement* self, GstWebRTCDataChannel* channel, gpointer user_data);
    static void onIceCandidateCallback(GstElement* self, guint mlineIndex, gchar* candidate, gpointer user_data);
    static void onOfferCreatedCallback(GstPromise* promise, gpointer user_data);
    static void onAnswerCreatedCallback(GstPromise* promise, gpointer user_data);

    struct WebRtcPromiseContext {
        Impl* impl = nullptr;
        std::string sessionId;
    };

    std::thread sessionMonitorThread;
    std::atomic<bool> sessionMonitorRunning = false;

    std::unique_ptr<Viewport::FrameCapture> frameCapture;
    std::thread frameSubmissionThread;
    std::atomic<bool> frameSubmissionRunning = false;

    // Frame submission
    std::mutex bufferMutex;
    std::condition_variable bufferCond;
    bool bufferProcessed = false;

    std::atomic<bool> forceKeyframe = false;
    std::chrono::time_point<std::chrono::steady_clock> initialFrameTime;
    std::chrono::time_point<std::chrono::steady_clock> lastKeyframeTime;

    static void OnBufferReleaseCallback(gpointer user_data);
    Result pushNewFrame(const void* data);
};

Instance::Remote::Remote(Viewport::Generic* viewport) {
    impl = std::make_shared<Impl>();
    impl->viewport = viewport;
    impl->viewportDevice = viewport ? viewport->device() : DeviceType::None;
}

Instance::Remote::~Remote() = default;

bool Instance::Remote::supported() const {
    return impl->supported();
}

Result Instance::Remote::create(const Config& config) {
    return impl->create(config);
}

Result Instance::Remote::destroy() {
    return impl->destroy();
}

bool Instance::Remote::started() const {
    return impl->started_;
}

Result Instance::Remote::captureFrame() {
    return impl->captureFrame();
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
    return impl->updateWaitlist();
}

Result Instance::Remote::updateSessions() {
    return impl->updateSessions();
}

Result Instance::Remote::approveClient(const std::string& code) {
    return impl->approveClient(code);
}

bool Instance::Remote::Impl::supported() const {
    return this->viewportDevice == DeviceType::Vulkan;
}

Result Instance::Remote::Impl::create(const Instance::Remote::Config& config) {
    JST_DEBUG("[REMOTE] Initializing remote streaming.");

    this->config = config;

    if (!supported()) {
        JST_ERROR("[REMOTE] Current backend ({}) is not supported for remote streaming.", this->viewportDevice);
        return Result::ERROR;
    }

    // Get viewport size
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (this->viewportDevice == DeviceType::Vulkan) {
        auto* vulkanAdapter = dynamic_cast<Viewport::Adapter<DeviceType::Vulkan>*>(this->viewport);
        if (vulkanAdapter) {
            this->size = vulkanAdapter->getSwapchainExtent();
        }
    }
#endif

    if (this->size.x == 0 || this->size.y == 0) {
        JST_ERROR("[REMOTE] Failed to get viewport size.");
        return Result::ERROR;
    }

    if (this->config.broker.empty()) {
        JST_ERROR("[REMOTE] Missing broker address.");
        return Result::ERROR;
    }

    JST_CHECK(this->createStream());
    JST_CHECK(this->createBroker());

    // Create frame capture
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (this->viewportDevice == DeviceType::Vulkan) {
        this->frameCapture = std::make_unique<Viewport::FrameCaptureVulkan>();
        JST_CHECK(this->frameCapture->create(this->viewport, inputMemoryDevice_));
    }
#endif

    if (!this->frameCapture) {
        JST_ERROR("[REMOTE] No frame capture available for device type.");
        return Result::ERROR;
    }

    // Start frame submission thread
    this->frameSubmissionRunning = true;
    this->frameSubmissionThread = std::thread([this]() {
        while (this->frameSubmissionRunning) {
            Tensor tensor;
            if (this->frameCapture->getFrameData(tensor) == Result::SUCCESS) {
                this->pushNewFrame(tensor.data());
                this->frameCapture->releaseFrame();
            }
        }
    });

    this->sessionMonitorRunning = true;
    this->sessionMonitorThread = std::thread([this]() {
        while (this->sessionMonitorRunning) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (!this->sessionMonitorRunning) break;
            updateWaitlist();
            updateSessions();
        }
    });

    this->started_ = true;
    JST_INFO("[REMOTE] Remote streaming started.");
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::destroy() {
    JST_DEBUG("[REMOTE] Destroying remote streaming.");

    this->sessionMonitorRunning = false;
    if (this->sessionMonitorThread.joinable()) {
        this->sessionMonitorThread.join();
    }

    this->frameSubmissionRunning = false;
    if (this->frameCapture) {
        this->frameCapture->stop();
    }
    if (this->frameSubmissionThread.joinable()) {
        this->frameSubmissionThread.join();
    }

    if (this->frameCapture) {
        this->frameCapture->destroy();
        this->frameCapture.reset();
    }

    JST_CHECK(this->destroyStream());
    JST_CHECK(this->destroyBroker());

    this->started_ = false;
    JST_INFO("[REMOTE] Remote streaming stopped.");
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::captureFrame() {
    if (this->frameCapture) {
        this->frameCapture->captureFrame();
    }
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::updateWaitlist() {
    if (!this->started_) {
        return Result::SUCCESS;
    }

    auto res = this->brokerClient->Get("/api/v1/remote/room/waitlist");
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

        this->waitlist_ = j["sessions"].get<std::vector<std::string>>();
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::updateSessions() {
    if (!this->started_) {
        return Result::SUCCESS;
    }

    auto res = this->brokerClient->Get("/api/v1/remote/room/active");
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
        this->clients_.clear();
        for (const auto& s : sessions) {
            this->clients_.push_back({s});
        }
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] JSON parse error '{}': {}", e.what(), res->body);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::approveClient(const std::string& code) {
    if (!this->started_) {
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

    for (auto& sessionId : this->waitlist_) {
        JST_DEBUG("[REMOTE] Candidate session: {}", sessionId);

        if (sessionId.ends_with(to_lower(code))) {
            JST_INFO("[REMOTE] Client authorization code '{}' approved.", code);

            auto json = nlohmann::json{{"sessionId", sessionId}};
            auto res = this->brokerClient->Post("/api/v1/remote/room/approval", json.dump(), "application/json");
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
        "rtp",
        "webrtc",
    };

    JST_CHECK(checkGstreamerPlugins(plugins));

    if (config.codec == Instance::Remote::CodecType::AV1) {
        JST_ERROR("[REMOTE] AV1 remote encoding is not implemented in the native WebRTC path.");
        return Result::ERROR;
    }

    std::vector<std::tuple<DeviceType, EncodingStrategyType, std::vector<std::string>>> combinations;

    if (config.codec == Instance::Remote::CodecType::H264) {
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (viewportDevice == DeviceType::Vulkan && Backend::State<DeviceType::CUDA>()->isAvailable()) {
            combinations.push_back({DeviceType::CUDA, EncodingStrategyType::HardwareNVENC, {"nvcodec"}});
            GST_DEBUG("[REMOTE] Checking for NVENC strategy support for h264.");
        }
#endif

        if (checkGstreamerPlugins({"applemedia"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("vtenc_h264_hw");
            if (factory) {
                combinations.push_back({DeviceType::CPU, EncodingStrategyType::HardwareVideoToolbox, {"applemedia"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[REMOTE] Checking for VideoToolbox strategy support for h264.");
            }
        }

        if (checkGstreamerPlugins({"video4linux2"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("v4l2h264enc");
            if (factory) {
                combinations.push_back({DeviceType::CPU, EncodingStrategyType::HardwareV4L2, {"video4linux2"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[REMOTE] Checking for V4L2 strategy support for h264.");
            }
        }

        combinations.push_back({DeviceType::CPU, EncodingStrategyType::Software, {"openh264"}});
    }

    if (config.codec == Instance::Remote::CodecType::VP8) {
        combinations.push_back({DeviceType::CPU, EncodingStrategyType::Software, {"vpx"}});
    }

    if (config.codec == Instance::Remote::CodecType::VP9) {
        combinations.push_back({DeviceType::CPU, EncodingStrategyType::Software, {"vpx"}});
    }

    bool requestedEncoderFound = false;

    for (const auto& [device, strategy, pluginList] : combinations) {
        if (config.encoder != Instance::Remote::EncoderType::Auto) {
            bool match = false;
            switch (config.encoder) {
                case Instance::Remote::EncoderType::Software:
                    match = (strategy == EncodingStrategyType::Software);
                    break;
                case Instance::Remote::EncoderType::NVENC:
                    match = (strategy == EncodingStrategyType::HardwareNVENC);
                    break;
                case Instance::Remote::EncoderType::V4L2:
                    match = (strategy == EncodingStrategyType::HardwareV4L2);
                    break;
                case Instance::Remote::EncoderType::VideoToolbox:
                    match = (strategy == EncodingStrategyType::HardwareVideoToolbox);
                    break;
                default:
                    break;
            }
            if (!match) {
                continue;
            }

            requestedEncoderFound = true;
        }

        if (checkGstreamerPlugins(pluginList, true) == Result::SUCCESS) {
            inputMemoryDevice_ = device;
            encodingStrategy = strategy;

            JST_INFO("[REMOTE] Using {} encoding with {} memory.", GetEncodingStrategyPrettyName(strategy),
                                                                   GetDevicePrettyName(device));

            return Result::SUCCESS;
        }
        JST_DEBUG("[REMOTE] Failed to find plugins: {}", pluginList);
    }

    if (config.encoder != Instance::Remote::EncoderType::Auto) {
        const std::string encoderName = Jetstream::GetRemoteEncoderPrettyName(config.encoder);
        const std::string codecName = Jetstream::GetRemoteCodecName(config.codec);

        if (!requestedEncoderFound) {
            JST_ERROR("[REMOTE] Requested encoder '{}' is not available for codec '{}' on this system.",
                      encoderName,
                      codecName);
        } else {
            JST_ERROR("[REMOTE] Requested encoder '{}' for codec '{}' matched a supported path, but the required GStreamer plugins were not available.",
                      encoderName,
                      codecName);
        }
    }

    JST_ERROR("[REMOTE] No encoding combination is available.");
    JST_ERROR("[REMOTE] This is typically caused by missing plugins.");
    return Result::ERROR;
}

Result Instance::Remote::Impl::destroyStream() {
    JST_DEBUG("[REMOTE] Destroying stream.");

    encodingStrategy = EncodingStrategyType::None;
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

    gchar *label = NULL; gint id = -1; gboolean negotiated = FALSE;
    g_object_get(channel, "label", &label, "id", &id, "negotiated", &negotiated, NULL);
    const std::string channelLabel = label ? label : "";

    if (channelLabel == "control") {
        JST_INFO("[REMOTE] Control data channel opened (id='{}', negotiated='{}').", id, negotiated);
        g_signal_connect(G_OBJECT(channel), "on-message-string", G_CALLBACK(onMessageCallback), user_data);
    } else {
        JST_INFO("[REMOTE] Dropping data channel opened from client (label='{}', id='{}', negotiated='{}')", channelLabel, id, negotiated);
        g_signal_emit_by_name(channel, "close");
    }

    g_free(label);
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

    if (encodingStrategy == EncodingStrategyType::None) {
        JST_ERROR("[REMOTE] No encoding strategy selected.");
        return Result::ERROR;
    }

    if (encodingStrategy == EncodingStrategyType::Software) {
        elements["rawparser"] = gst_element_factory_make("rawvideoparse", "rawparser");
        elementOrder.push_back("rawparser");

        g_object_set(elements["rawparser"], "use-sink-caps", 0, nullptr);
        g_object_set(elements["rawparser"], "format", 12, nullptr);
        g_object_set(elements["rawparser"], "width", static_cast<int>(size.x), nullptr);
        g_object_set(elements["rawparser"], "height", static_cast<int>(size.y), nullptr);

        elements["convert"] = gst_element_factory_make("videoconvert", "convert");
        elementOrder.push_back("convert");

        switch(config.codec) {
            case Instance::Remote::CodecType::H264: {
                elements["encoder"] = encoder = gst_element_factory_make("openh264enc", "encoder");
                elementOrder.push_back("encoder");

                gst_util_set_object_arg(G_OBJECT(elements["encoder"]), "rate-control", "bitrate");
                gst_util_set_object_arg(G_OBJECT(elements["encoder"]), "usage-type", "screen");
                g_object_set(elements["encoder"], "bitrate", 25*1024*1024, nullptr);
                g_object_set(elements["encoder"], "max-bitrate", 25*1024*1024, nullptr);

                elements["hwcaps"] = gst_element_factory_make("capsfilter", "hwcaps");
                elementOrder.push_back("hwcaps");

                GstCaps* hwcaps = gst_caps_new_simple("video/x-h264",
                                                      "profile", G_TYPE_STRING, "constrained-baseline",
                                                      nullptr);
                g_object_set(elements["hwcaps"], "caps", hwcaps, nullptr);
                gst_caps_unref(hwcaps);

                elements["parser"] = gst_element_factory_make("h264parse", "parser");
                elementOrder.push_back("parser");

                g_object_set(elements["parser"], "config-interval", 1, nullptr);
                break;
            }
            case Instance::Remote::CodecType::VP8:
                elements["encoder"] = encoder = gst_element_factory_make("vp8enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "target-bitrate", 25*1024*1024, nullptr);
                break;
            case Instance::Remote::CodecType::VP9:
                elements["encoder"] = encoder = gst_element_factory_make("vp9enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "target-bitrate", 25*1024*1024, nullptr);
                break;
            default:
                JST_ERROR("[REMOTE] Unsupported codec for software encoding.");
                return Result::ERROR;
        }
    }

    if (encodingStrategy == EncodingStrategyType::HardwareNVENC) {
        switch(config.codec) {
            case Instance::Remote::CodecType::H264:
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

    if (encodingStrategy == EncodingStrategyType::HardwareV4L2) {
        elements["rawparser"] = gst_element_factory_make("rawvideoparse", "rawparser");
        elementOrder.push_back("rawparser");

        g_object_set(elements["rawparser"], "use-sink-caps", 0, nullptr);
        g_object_set(elements["rawparser"], "format", 12, nullptr);
        g_object_set(elements["rawparser"], "width", static_cast<int>(size.x), nullptr);
        g_object_set(elements["rawparser"], "height", static_cast<int>(size.y), nullptr);

        elements["convert"] = gst_element_factory_make("videoconvert", "convert");
        elementOrder.push_back("convert");

        switch(config.codec) {
            case Instance::Remote::CodecType::H264: {
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

    if (encodingStrategy == EncodingStrategyType::HardwareVideoToolbox) {
        elements["rawparser"] = gst_element_factory_make("rawvideoparse", "rawparser");
        elementOrder.push_back("rawparser");

        g_object_set(elements["rawparser"], "use-sink-caps", 0, nullptr);
        g_object_set(elements["rawparser"], "format", 12, nullptr);
        g_object_set(elements["rawparser"], "width", static_cast<int>(size.x), nullptr);
        g_object_set(elements["rawparser"], "height", static_cast<int>(size.y), nullptr);

        elements["convert"] = gst_element_factory_make("videoconvert", "convert");
        elementOrder.push_back("convert");

        switch(config.codec) {
            case Instance::Remote::CodecType::H264: {
                elements["encoder"] = encoder = gst_element_factory_make("vtenc_h264_hw", "encoder");
                elementOrder.push_back("encoder");

                gst_util_set_object_arg(G_OBJECT(elements["encoder"]), "rate-control", "cbr");
                g_object_set(elements["encoder"], "bitrate", 25*1024, nullptr);
                g_object_set(elements["encoder"], "realtime", true, nullptr);
                g_object_set(elements["encoder"], "allow-frame-reordering", false, nullptr);
                g_object_set(elements["encoder"], "max-keyframe-interval", static_cast<int>(config.framerate), nullptr);

                elements["hwcaps"] = gst_element_factory_make("capsfilter", "hwcaps");
                elementOrder.push_back("hwcaps");

                GstCaps* hwcaps = gst_caps_new_simple("video/x-h264",
                                                      "profile", G_TYPE_STRING, "baseline",
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

    // 04. Setup stream tee. WebRTC peer branches are added per session.

    elements["tee"] = tee = gst_element_factory_make("tee", "tee");
    elementOrder.push_back("tee");

    if (!elements["tee"]) {
        JST_ERROR("[REMOTE] Failed to create gstreamer element 'tee'.");
        return Result::ERROR;
    }

    g_object_set(elements["tee"], "allow-not-linked", true, nullptr);

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

    if (startSignaller() != Result::SUCCESS) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline = nullptr;
        source = nullptr;
        encoder = nullptr;
        tee = nullptr;
        return Result::ERROR;
    }

    initialFrameTime = std::chrono::steady_clock::now();
    lastKeyframeTime = initialFrameTime;
    forceKeyframe.store(true);
    streaming = true;

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::stopStream() {
    JST_DEBUG("[REMOTE] Stopping stream.");

    JST_CHECK(stopSignaller());
    destroyAllWebRtcSessions();

    if (streaming) {
        std::lock_guard<std::mutex> lock(streamMutex);
        streaming = false;

        gst_element_send_event(pipeline, gst_event_new_eos());

        GstBus* bus = gst_element_get_bus(pipeline);
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_SECOND, GST_MESSAGE_EOS);
        if (msg) {
            gst_message_unref(msg);
        }
        gst_object_unref(bus);

        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline = nullptr;
        source = nullptr;
        encoder = nullptr;
        tee = nullptr;
    }

    return Result::SUCCESS;
}

GstElement* Instance::Remote::Impl::createPayloader() {
    GstElement* element = nullptr;

    switch(config.codec) {
        case Instance::Remote::CodecType::H264:
            element = gst_element_factory_make("rtph264pay", nullptr);
            if (element) {
                g_object_set(element, "pt", 96, nullptr);
                g_object_set(element, "config-interval", -1, nullptr);
                gst_util_set_object_arg(G_OBJECT(element), "aggregate-mode", "zero-latency");
            }
            break;
        case Instance::Remote::CodecType::VP8:
            element = gst_element_factory_make("rtpvp8pay", nullptr);
            if (element) {
                g_object_set(element, "pt", 96, nullptr);
            }
            break;
        case Instance::Remote::CodecType::VP9:
            element = gst_element_factory_make("rtpvp9pay", nullptr);
            if (element) {
                g_object_set(element, "pt", 96, nullptr);
            }
            break;
        default:
            JST_ERROR("[REMOTE] Unsupported codec for WebRTC streaming.");
            break;
    }

    return element;
}

GstElement* Instance::Remote::Impl::createRtpCapsFilter() {
    GstElement* element = gst_element_factory_make("capsfilter", nullptr);
    if (!element) {
        return nullptr;
    }

    GstCaps* caps = nullptr;
    switch(config.codec) {
        case Instance::Remote::CodecType::H264:
            caps = gst_caps_new_simple("application/x-rtp",
                                       "media", G_TYPE_STRING, "video",
                                       "encoding-name", G_TYPE_STRING, "H264",
                                       "payload", G_TYPE_INT, 96,
                                       "clock-rate", G_TYPE_INT, 90000,
                                       "packetization-mode", G_TYPE_STRING, "1",
                                       nullptr);
            break;
        case Instance::Remote::CodecType::VP8:
            caps = gst_caps_new_simple("application/x-rtp",
                                       "media", G_TYPE_STRING, "video",
                                       "encoding-name", G_TYPE_STRING, "VP8",
                                       "payload", G_TYPE_INT, 96,
                                       "clock-rate", G_TYPE_INT, 90000,
                                       nullptr);
            break;
        case Instance::Remote::CodecType::VP9:
            caps = gst_caps_new_simple("application/x-rtp",
                                       "media", G_TYPE_STRING, "video",
                                       "encoding-name", G_TYPE_STRING, "VP9",
                                       "payload", G_TYPE_INT, 96,
                                       "clock-rate", G_TYPE_INT, 90000,
                                       nullptr);
            break;
        default:
            JST_ERROR("[REMOTE] Unsupported codec for WebRTC streaming.");
            gst_object_unref(element);
            return nullptr;
    }

    g_object_set(element, "caps", caps, nullptr);
    gst_caps_unref(caps);

    return element;
}

GstElement* Instance::Remote::Impl::refSessionWebrtc(const std::string& sessionId) {
    std::lock_guard<std::mutex> lock(sessionsMutex);
    auto it = sessions.find(sessionId);
    if (it == sessions.end() || !it->second->webrtc) {
        return nullptr;
    }

    return GST_ELEMENT(gst_object_ref(it->second->webrtc));
}

Result Instance::Remote::Impl::createWebRtcSession(const std::string& sessionId, const std::string& peerId) {
    std::lock_guard<std::mutex> lock(sessionsMutex);

    if (sessions.contains(sessionId)) {
        return Result::SUCCESS;
    }

    if (!pipeline || !tee) {
        JST_ERROR("[REMOTE] Can't create WebRTC session without an active stream.");
        return Result::ERROR;
    }

    auto session = std::make_unique<WebRtcSession>();
    session->sessionId = sessionId;
    session->peerId = peerId;
    session->queue = gst_element_factory_make("queue", nullptr);
    session->payloader = createPayloader();
    session->rtpCaps = createRtpCapsFilter();
    session->webrtc = gst_element_factory_make("webrtcbin", nullptr);

    std::lock_guard<std::mutex> streamLock(streamMutex);

    auto removeElement = [this](GstElement*& element) {
        if (!element) {
            return;
        }

        gst_element_set_state(element, GST_STATE_NULL);
        if (pipeline && GST_OBJECT_PARENT(element)) {
            gst_bin_remove(GST_BIN(pipeline), element);
        } else {
            gst_object_unref(element);
        }
        element = nullptr;
    };

    auto fail = [&]() {
        if (session->teeSrcPad) {
            if (tee) {
                gst_element_release_request_pad(tee, session->teeSrcPad);
            }
            gst_object_unref(session->teeSrcPad);
            session->teeSrcPad = nullptr;
        }

        if (session->webrtcSinkPad) {
            if (session->webrtc) {
                gst_element_release_request_pad(session->webrtc, session->webrtcSinkPad);
            }
            gst_object_unref(session->webrtcSinkPad);
            session->webrtcSinkPad = nullptr;
        }

        removeElement(session->webrtc);
        removeElement(session->rtpCaps);
        removeElement(session->payloader);
        removeElement(session->queue);
        return Result::ERROR;
    };

    if (!session->queue || !session->payloader || !session->rtpCaps || !session->webrtc) {
        JST_ERROR("[REMOTE] Failed to create WebRTC session elements.");
        return fail();
    }

    g_object_set(session->queue,
                 "leaky", 2,
                 "max-size-buffers", 2u,
                 "max-size-bytes", 0u,
                 "max-size-time", static_cast<guint64>(0),
                 nullptr);
    g_object_set(session->webrtc, "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, nullptr);

    if (!gst_bin_add(GST_BIN(pipeline), session->queue) ||
        !gst_bin_add(GST_BIN(pipeline), session->payloader) ||
        !gst_bin_add(GST_BIN(pipeline), session->rtpCaps) ||
        !gst_bin_add(GST_BIN(pipeline), session->webrtc)) {
        JST_ERROR("[REMOTE] Failed to add WebRTC session elements to pipeline.");
        return fail();
    }

    if (!gst_element_link(session->queue, session->payloader) ||
        !gst_element_link(session->payloader, session->rtpCaps)) {
        JST_ERROR("[REMOTE] Failed to link WebRTC session RTP elements.");
        return fail();
    }

    GstPad* rtpCapsSrcPad = gst_element_get_static_pad(session->rtpCaps, "src");
    session->webrtcSinkPad = gst_element_request_pad_simple(session->webrtc, "sink_%u");
    if (!rtpCapsSrcPad || !session->webrtcSinkPad) {
        JST_ERROR("[REMOTE] Failed to create WebRTC RTP link pads.");
        if (rtpCapsSrcPad) gst_object_unref(rtpCapsSrcPad);
        return fail();
    }

    const GstPadLinkReturn webrtcLinkReturn = gst_pad_link(rtpCapsSrcPad, session->webrtcSinkPad);
    gst_object_unref(rtpCapsSrcPad);
    if (webrtcLinkReturn != GST_PAD_LINK_OK) {
        JST_ERROR("[REMOTE] Failed to link RTP payloader to WebRTC element.");
        return fail();
    }

    session->teeSrcPad = gst_element_request_pad_simple(tee, "src_%u");
    GstPad* queueSinkPad = gst_element_get_static_pad(session->queue, "sink");
    if (!session->teeSrcPad || !queueSinkPad) {
        JST_ERROR("[REMOTE] Failed to create WebRTC tee link pads.");
        if (queueSinkPad) gst_object_unref(queueSinkPad);
        return fail();
    }

    const GstPadLinkReturn teeLinkReturn = gst_pad_link(session->teeSrcPad, queueSinkPad);
    gst_object_unref(queueSinkPad);
    if (teeLinkReturn != GST_PAD_LINK_OK) {
        JST_ERROR("[REMOTE] Failed to link stream tee to WebRTC session queue.");
        return fail();
    }

    auto* context = new WebRtcSignalContext{this, sessionId};
    session->iceHandler = g_signal_connect_data(G_OBJECT(session->webrtc),
                                                "on-ice-candidate",
                                                G_CALLBACK(onIceCandidateCallback),
                                                context,
                                                [](gpointer data, GClosure*) {
                                                    delete reinterpret_cast<WebRtcSignalContext*>(data);
                                                },
                                                GConnectFlags(0));
    session->channelHandler = g_signal_connect(G_OBJECT(session->webrtc),
                                               "on-data-channel",
                                               G_CALLBACK(onChannelCallback),
                                               this);

    if (!gst_element_sync_state_with_parent(session->queue) ||
        !gst_element_sync_state_with_parent(session->payloader) ||
        !gst_element_sync_state_with_parent(session->rtpCaps) ||
        !gst_element_sync_state_with_parent(session->webrtc)) {
        JST_ERROR("[REMOTE] Failed to start WebRTC session elements.");
        return fail();
    }

    sessions[sessionId] = std::move(session);
    forceKeyframe.store(true);

    return Result::SUCCESS;
}

void Instance::Remote::Impl::destroyWebRtcSession(const std::string& sessionId) {
    std::unique_ptr<WebRtcSession> session;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        auto it = sessions.find(sessionId);
        if (it == sessions.end()) {
            return;
        }

        session = std::move(it->second);
        sessions.erase(it);
    }

    JST_INFO("[REMOTE] Destroying WebRTC session '{}' with peer '{}'.", session->sessionId, session->peerId);

    std::lock_guard<std::mutex> streamLock(streamMutex);

    if (session->webrtc) {
        if (session->iceHandler) {
            g_signal_handler_disconnect(G_OBJECT(session->webrtc), session->iceHandler);
            session->iceHandler = 0;
        }
        if (session->channelHandler) {
            g_signal_handler_disconnect(G_OBJECT(session->webrtc), session->channelHandler);
            session->channelHandler = 0;
        }
    }

    if (session->queue) gst_element_set_state(session->queue, GST_STATE_NULL);
    if (session->payloader) gst_element_set_state(session->payloader, GST_STATE_NULL);
    if (session->rtpCaps) gst_element_set_state(session->rtpCaps, GST_STATE_NULL);
    if (session->webrtc) gst_element_set_state(session->webrtc, GST_STATE_NULL);

    if (session->teeSrcPad) {
        GstPad* queueSinkPad = session->queue ? gst_element_get_static_pad(session->queue, "sink") : nullptr;
        if (queueSinkPad) {
            gst_pad_unlink(session->teeSrcPad, queueSinkPad);
            gst_object_unref(queueSinkPad);
        }
        if (tee) {
            gst_element_release_request_pad(tee, session->teeSrcPad);
        }
        gst_object_unref(session->teeSrcPad);
        session->teeSrcPad = nullptr;
    }

    if (session->webrtcSinkPad) {
        if (session->webrtc) {
            gst_element_release_request_pad(session->webrtc, session->webrtcSinkPad);
        }
        gst_object_unref(session->webrtcSinkPad);
        session->webrtcSinkPad = nullptr;
    }

    auto removeElement = [this](GstElement*& element) {
        if (!element) {
            return;
        }

        if (pipeline && GST_OBJECT_PARENT(element)) {
            gst_bin_remove(GST_BIN(pipeline), element);
        }
        element = nullptr;
    };

    removeElement(session->webrtc);
    removeElement(session->rtpCaps);
    removeElement(session->payloader);
    removeElement(session->queue);
}

void Instance::Remote::Impl::destroyAllWebRtcSessions() {
    std::vector<std::string> sessionIds;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        for (const auto& [sessionId, session] : sessions) {
            (void)session;
            sessionIds.push_back(sessionId);
        }
    }

    for (const auto& sessionId : sessionIds) {
        destroyWebRtcSession(sessionId);
    }
}

//
// Signalling
//

Result Instance::Remote::Impl::startSignaller() {
    JST_DEBUG("[REMOTE] Starting WebRTC signaller.");

    signallerClient = std::make_unique<httplib::ws::WebSocketClient>(signallerUrl);
    signallerClient->set_write_timeout(1);
    signallerClient->set_tcp_nodelay(true);

    if (!signallerClient->is_valid() || !signallerClient->connect()) {
        JST_ERROR("[REMOTE] Failed to connect to signaller '{}'.", signallerUrl);
        signallerClient.reset();
        return Result::ERROR;
    }

    signallerRunning = true;
    signallerThread = std::thread([this]() { signallerLoop(); });

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::stopSignaller() {
    signallerRunning = false;

    {
        std::lock_guard<std::mutex> lock(signallerMutex);
        if (signallerClient) {
            signallerClient->close();
        }
    }

    if (signallerThread.joinable()) {
        signallerThread.join();
    }

    {
        std::lock_guard<std::mutex> lock(signallerMutex);
        signallerClient.reset();
        producerPeerId.clear();
    }

    return Result::SUCCESS;
}

void Instance::Remote::Impl::signallerLoop() {
    while (signallerRunning) {
        std::string payload;
        httplib::ws::ReadResult result = httplib::ws::Fail;
        httplib::ws::WebSocketClient* client = nullptr;

        {
            std::lock_guard<std::mutex> lock(signallerMutex);
            if (!signallerClient || !signallerClient->is_open()) {
                break;
            }
            client = signallerClient.get();
        }

        if (client) {
            result = client->read(payload);
        }

        if (result == httplib::ws::Text) {
            handleSignallerMessage(payload);
        } else if (result == httplib::ws::Binary) {
            JST_WARN("[REMOTE] Ignoring binary signaller message.");
        } else if (signallerRunning) {
            JST_ERROR("[REMOTE] Signaller connection closed.");
            break;
        }
    }
}

void Instance::Remote::Impl::handleSignallerMessage(const std::string& payload) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(payload);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Bad signaller JSON: {} (payload='{}')", e.what(), payload);
        return;
    }

    const std::string type = j.value("type", "");
    if (type == "welcome") {
        {
            std::lock_guard<std::mutex> lock(signallerMutex);
            producerPeerId = j.value("peerId", "");
        }

        nlohmann::json status = {
            {"type", "setPeerStatus"},
            {"peerId", j.value("peerId", "")},
            {"roles", nlohmann::json::array({"producer"})},
            {"meta", {{"token", producerToken}}},
        };
        (void)sendSignallerMessage(status);
        return;
    }

    if (type == "startSession") {
        handleStartSession(j);
        return;
    }

    if (type == "peer") {
        handlePeerMessage(j);
        return;
    }

    if (type == "endSession") {
        handleEndSession(j);
        return;
    }

    if (type == "error") {
        JST_ERROR("[REMOTE] Signaller error: {}", j.value("details", "unknown"));
        return;
    }

    JST_TRACE("[REMOTE] Ignoring signaller message type '{}'.", type);
}

void Instance::Remote::Impl::handleStartSession(const nlohmann::json& j) {
    const std::string sessionId = j.value("sessionId", "");
    const std::string peerId = j.value("peerId", "");
    if (sessionId.empty() || peerId.empty()) {
        JST_ERROR("[REMOTE] Invalid startSession message: {}", j.dump());
        return;
    }

    if (createWebRtcSession(sessionId, peerId) != Result::SUCCESS) {
        (void)sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}});
        return;
    }

    JST_INFO("[REMOTE] Starting WebRTC session '{}' with peer '{}'.", sessionId, peerId);

    if (j.contains("offer") && !j["offer"].is_null()) {
        const std::string offer = j["offer"].get<std::string>();
        if (applyRemoteDescription(sessionId, "offer", offer) != Result::SUCCESS) {
            destroyWebRtcSession(sessionId);
            (void)sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}});
        }
        return;
    }

    createControlChannel(sessionId);

    auto* context = new WebRtcPromiseContext{this, sessionId};
    GstPromise* promise = gst_promise_new_with_change_func(onOfferCreatedCallback, context, [](gpointer data) {
        delete reinterpret_cast<WebRtcPromiseContext*>(data);
    });

    GstElement* sessionWebrtc = refSessionWebrtc(sessionId);
    if (!sessionWebrtc) {
        gst_promise_unref(promise);
        return;
    }

    g_signal_emit_by_name(G_OBJECT(sessionWebrtc), "create-offer", nullptr, promise);
    gst_object_unref(sessionWebrtc);
}

void Instance::Remote::Impl::handlePeerMessage(const nlohmann::json& j) {
    const std::string sessionId = j.value("sessionId", "");
    if (sessionId.empty()) {
        JST_ERROR("[REMOTE] Peer message without sessionId: {}", j.dump());
        return;
    }

    if (j.contains("sdp")) {
        const auto& sdp = j["sdp"];
        if (applyRemoteDescription(sessionId, sdp.value("type", ""), sdp.value("sdp", "")) != Result::SUCCESS) {
            destroyWebRtcSession(sessionId);
            (void)sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}});
        }
        return;
    }

    if (j.contains("ice")) {
        const auto& ice = j["ice"];
        const guint mlineIndex = ice.value("sdpMLineIndex", ice.value("sdp_m_line_index", 0));
        const std::string candidate = ice.value("candidate", "");
        if (!candidate.empty()) {
            GstElement* sessionWebrtc = refSessionWebrtc(sessionId);
            if (sessionWebrtc) {
                g_signal_emit_by_name(G_OBJECT(sessionWebrtc), "add-ice-candidate", mlineIndex, candidate.c_str());
                gst_object_unref(sessionWebrtc);
            }
        }
        return;
    }
}

void Instance::Remote::Impl::handleEndSession(const nlohmann::json& j) {
    const std::string sessionId = j.value("sessionId", "");
    if (!sessionId.empty()) {
        JST_INFO("[REMOTE] WebRTC session '{}' ended.", sessionId);
        destroyWebRtcSession(sessionId);
    }
}

bool Instance::Remote::Impl::sendSignallerMessage(const nlohmann::json& j) {
    const std::string payload = j.dump();
    std::lock_guard<std::mutex> lock(signallerMutex);
    if (!signallerClient || !signallerClient->is_open()) {
        return false;
    }
    return signallerClient->send(payload);
}

void Instance::Remote::Impl::sendSessionDescription(const std::string& sessionId,
                                                    const GstWebRTCSessionDescription* desc) {
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        if (!sessions.contains(sessionId)) {
            return;
        }
    }

    if (!desc || !desc->sdp) {
        JST_ERROR("[REMOTE] Can't send empty WebRTC session description.");
        return;
    }

    gchar* sdpText = gst_sdp_message_as_text(desc->sdp);
    if (!sdpText) {
        JST_ERROR("[REMOTE] Failed to serialize WebRTC session description.");
        return;
    }

    const char* type = desc->type == GST_WEBRTC_SDP_TYPE_OFFER ? "offer" : "answer";
    nlohmann::json msg = {
        {"type", "peer"},
        {"sessionId", sessionId},
        {"sdp", {{"type", type}, {"sdp", std::string(sdpText)}}},
    };

    g_free(sdpText);
    if (!sendSignallerMessage(msg)) {
        JST_ERROR("[REMOTE] Failed to send WebRTC session description.");
    }
}

void Instance::Remote::Impl::sendIceCandidate(const std::string& sessionId, guint mlineIndex, const gchar* candidate) {
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        if (!sessions.contains(sessionId)) {
            return;
        }
    }

    if (sessionId.empty() || !candidate) {
        return;
    }

    nlohmann::json msg = {
        {"type", "peer"},
        {"sessionId", sessionId},
        {"ice", {{"candidate", std::string(candidate)}, {"sdpMLineIndex", mlineIndex}}},
    };

    if (!sendSignallerMessage(msg)) {
        JST_ERROR("[REMOTE] Failed to send WebRTC ICE candidate.");
    }
}

Result Instance::Remote::Impl::applyRemoteDescription(const std::string& sessionId,
                                                      const std::string& type,
                                                      const std::string& sdpText) {
    if (type.empty() || sdpText.empty()) {
        JST_ERROR("[REMOTE] Invalid WebRTC SDP message.");
        return Result::ERROR;
    }

    if (type != "offer" && type != "answer") {
        JST_ERROR("[REMOTE] Unsupported WebRTC SDP type '{}'.", type);
        return Result::ERROR;
    }

    GstElement* sessionWebrtc = refSessionWebrtc(sessionId);
    if (!sessionWebrtc) {
        JST_ERROR("[REMOTE] WebRTC session '{}' does not exist.", sessionId);
        return Result::ERROR;
    }

    GstSDPMessage* sdp = nullptr;
    if (gst_sdp_message_new(&sdp) != GST_SDP_OK) {
        JST_ERROR("[REMOTE] Failed to allocate WebRTC SDP message.");
        gst_object_unref(sessionWebrtc);
        return Result::ERROR;
    }

    const auto* data = reinterpret_cast<const guint8*>(sdpText.data());
    if (gst_sdp_message_parse_buffer(data, sdpText.size(), sdp) != GST_SDP_OK) {
        JST_ERROR("[REMOTE] Failed to parse WebRTC SDP message.");
        gst_sdp_message_free(sdp);
        gst_object_unref(sessionWebrtc);
        return Result::ERROR;
    }

    const GstWebRTCSDPType descType = type == "offer" ? GST_WEBRTC_SDP_TYPE_OFFER : GST_WEBRTC_SDP_TYPE_ANSWER;
    GstWebRTCSessionDescription* desc = gst_webrtc_session_description_new(descType, sdp);
    GstPromise* promise = gst_promise_new();
    g_signal_emit_by_name(G_OBJECT(sessionWebrtc), "set-remote-description", desc, promise);
    gst_promise_interrupt(promise);
    gst_promise_unref(promise);

    if (descType == GST_WEBRTC_SDP_TYPE_OFFER) {
        auto* context = new WebRtcPromiseContext{this, sessionId};
        GstPromise* answerPromise = gst_promise_new_with_change_func(onAnswerCreatedCallback, context, [](gpointer data) {
            delete reinterpret_cast<WebRtcPromiseContext*>(data);
        });
        g_signal_emit_by_name(G_OBJECT(sessionWebrtc), "create-answer", nullptr, answerPromise);
    }

    gst_webrtc_session_description_free(desc);
    gst_object_unref(sessionWebrtc);
    return Result::SUCCESS;
}

void Instance::Remote::Impl::createControlChannel(const std::string& sessionId) {
    GstElement* sessionWebrtc = refSessionWebrtc(sessionId);
    if (!sessionWebrtc) {
        JST_ERROR("[REMOTE] WebRTC session '{}' does not exist.", sessionId);
        return;
    }

    GstWebRTCDataChannel* channel = nullptr;
    g_signal_emit_by_name(G_OBJECT(sessionWebrtc), "create-data-channel", "control", nullptr, &channel);
    if (!channel) {
        JST_ERROR("[REMOTE] Failed to create control data channel.");
        gst_object_unref(sessionWebrtc);
        return;
    }
    g_signal_connect(G_OBJECT(channel), "on-message-string", G_CALLBACK(onMessageCallback), this);
    g_object_unref(channel);
    gst_object_unref(sessionWebrtc);
}

void Instance::Remote::Impl::onIceCandidateCallback(GstElement* self,
                                                    guint mlineIndex,
                                                    gchar* candidate,
                                                    gpointer user_data) {
    (void)self;
    auto* context = reinterpret_cast<WebRtcSignalContext*>(user_data);
    context->impl->sendIceCandidate(context->sessionId, mlineIndex, candidate);
}

void Instance::Remote::Impl::onOfferCreatedCallback(GstPromise* promise, gpointer user_data) {
    auto* context = reinterpret_cast<WebRtcPromiseContext*>(user_data);
    GstStructure const* reply = gst_promise_get_reply(promise);
    GstWebRTCSessionDescription* offer = nullptr;
    if (!reply) {
        JST_ERROR("[REMOTE] Failed to create WebRTC offer.");
        gst_promise_unref(promise);
        return;
    }

    gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, nullptr);

    if (!offer) {
        JST_ERROR("[REMOTE] Failed to create WebRTC offer.");
        gst_promise_unref(promise);
        return;
    }

    GstElement* sessionWebrtc = context->impl->refSessionWebrtc(context->sessionId);
    if (!sessionWebrtc) {
        gst_webrtc_session_description_free(offer);
        gst_promise_unref(promise);
        return;
    }

    GstPromise* localPromise = gst_promise_new();
    g_signal_emit_by_name(G_OBJECT(sessionWebrtc), "set-local-description", offer, localPromise);
    gst_promise_interrupt(localPromise);
    gst_promise_unref(localPromise);

    context->impl->sendSessionDescription(context->sessionId, offer);
    gst_webrtc_session_description_free(offer);
    gst_object_unref(sessionWebrtc);
    gst_promise_unref(promise);
}

void Instance::Remote::Impl::onAnswerCreatedCallback(GstPromise* promise, gpointer user_data) {
    auto* context = reinterpret_cast<WebRtcPromiseContext*>(user_data);
    GstStructure const* reply = gst_promise_get_reply(promise);
    GstWebRTCSessionDescription* answer = nullptr;
    if (!reply) {
        JST_ERROR("[REMOTE] Failed to create WebRTC answer.");
        gst_promise_unref(promise);
        return;
    }

    gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, nullptr);

    if (!answer) {
        JST_ERROR("[REMOTE] Failed to create WebRTC answer.");
        gst_promise_unref(promise);
        return;
    }

    GstElement* sessionWebrtc = context->impl->refSessionWebrtc(context->sessionId);
    if (!sessionWebrtc) {
        gst_webrtc_session_description_free(answer);
        gst_promise_unref(promise);
        return;
    }

    GstPromise* localPromise = gst_promise_new();
    g_signal_emit_by_name(G_OBJECT(sessionWebrtc), "set-local-description", answer, localPromise);
    gst_promise_interrupt(localPromise);
    gst_promise_unref(localPromise);

    context->impl->sendSessionDescription(context->sessionId, answer);
    gst_webrtc_session_description_free(answer);
    gst_object_unref(sessionWebrtc);
    gst_promise_unref(promise);
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
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        if (sessions.empty()) {
            return Result::SUCCESS;
        }
    }

    {
        std::lock_guard<std::mutex> lock(streamMutex);

        if (!streaming || !source) {
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

        const bool keyframeRequested = forceKeyframe.exchange(false);
        if ((elapsedSinceLastKeyframe.count() > 1) || keyframeRequested) {
            GstEvent* force_key_unit_event = gst_video_event_new_downstream_force_key_unit(
                GST_CLOCK_TIME_NONE,
                GST_CLOCK_TIME_NONE,
                GST_CLOCK_TIME_NONE,
                TRUE,
                0
            );

            gst_element_send_event(encoder, force_key_unit_event);

            lastKeyframeTime = currentFrameTime;
        }

        if (gst_app_src_push_buffer(GST_APP_SRC(source), buffer) != GST_FLOW_OK) {
            JST_ERROR("[REMOTE] Failed to push buffer to gstreamer pipeline.");
            return Result::ERROR;
        }
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

const char* Instance::Remote::Impl::GetEncodingStrategyPrettyName(const EncodingStrategyType& strategy) {
    switch (strategy) {
        case EncodingStrategyType::None:
            return "None";
        case EncodingStrategyType::Software:
            return "Software";
        case EncodingStrategyType::HardwareNVENC:
            return "Hardware NVIDIA (NVENC)";
        case EncodingStrategyType::HardwareV4L2:
            return "Hardware Linux (V4L2)";
        case EncodingStrategyType::HardwareVideoToolbox:
            return "Hardware Apple (VideoToolbox)";
        default:
            return "Unknown";
    }
}

}  // namespace Jetstream
