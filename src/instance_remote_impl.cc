#include "instance_remote_impl.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/capture.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/viewport/capture/vulkan.hh"
#endif

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>

#include <gst/app/gstappsrc.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video-event.h>
#include <gst/gststructure.h>

extern "C" void gst_init_static_plugins(void);

namespace Jetstream {

Instance::Remote::Remote(Viewport::Generic* viewport) {
    impl = std::make_shared<Impl>();
    impl->viewport = viewport;
    impl->viewportDevice = viewport ? viewport->device() : DeviceType::None;
}

Instance::Remote::~Remote() = default;

bool Instance::Remote::supported() const {
    return impl->supported();
}

std::vector<Instance::Remote::EncoderType> Instance::Remote::available(CodecType codec) {
    return impl->available(codec);
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

std::vector<Instance::Remote::ClientInfo> Instance::Remote::clients() const {
    std::lock_guard<std::mutex> lock(impl->remoteStateMutex);
    return impl->clients_;
}

std::vector<std::string> Instance::Remote::waitlist() const {
    std::lock_guard<std::mutex> lock(impl->remoteStateMutex);
    return impl->waitlist_;
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

    this->started_ = true;
    JST_INFO("[REMOTE] Remote streaming started.");
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::destroy() {
    JST_DEBUG("[REMOTE] Destroying remote streaming.");

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

Result Instance::Remote::Impl::approveClient(const std::string& code) {
    if (!this->started_) {
        JST_ERROR("[REMOTE] Can't approve client when session is not started.");
        return Result::ERROR;
    }

    auto to_lower = [](std::string_view s) {
        std::string out(s);
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return out;
    };

    std::vector<std::string> waitlist;
    {
        std::lock_guard<std::mutex> lock(remoteStateMutex);
        waitlist = waitlist_;
    }
    for (const auto& sessionId : waitlist) {
        JST_DEBUG("[REMOTE] Candidate session: {}", sessionId);

        if (sessionId.ends_with(to_lower(code))) {
            JST_INFO("[REMOTE] Client authorization code '{}' approved.", code);

            if (!sendSignallerMessage({{"type", "approveSession"}, {"sessionId", sessionId}})) {
                JST_ERROR("[REMOTE] Failed to send client approval.");
                return Result::ERROR;
            }

            return Result::SUCCESS;
        }
    }

    JST_ERROR("[REMOTE] Client authorization code '{}' not found.", code);
    return Result::ERROR;
}

//
// Stream
//

std::vector<Instance::Remote::EncoderType> Instance::Remote::Impl::available(CodecType codec) {
    if (!supported()) {
        return {};
    }

    auto cacheIt = availableEncoderCache.find(codec);
    if (cacheIt != availableEncoderCache.end()) {
        return cacheIt->second;
    }

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    gst_init_static_plugins();

    std::vector<EncoderType> encoders;
    const auto addEncoder = [&](EncoderType encoder) {
        if (std::find(encoders.begin(), encoders.end(), encoder) == encoders.end()) {
            encoders.push_back(encoder);
        }
    };

    for (const auto& combination : encodingCombinations(codec)) {
        if (checkGstreamerPlugins(combination.plugins, true) == Result::SUCCESS) {
            addEncoder(EncoderFromEncodingStrategy(combination.strategy));
        }
    }

    if (!encoders.empty()) {
        encoders.insert(encoders.begin(), EncoderType::Auto);
    }

    availableEncoderCache[codec] = encoders;
    return encoders;
}

std::vector<Instance::Remote::Impl::EncodingCombination> Instance::Remote::Impl::encodingCombinations(CodecType codec) {
    std::vector<EncodingCombination> combinations;

    if (codec == CodecType::H264) {
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (viewportDevice == DeviceType::Vulkan && Backend::State<DeviceType::CUDA>()->isAvailable()) {
            combinations.push_back({DeviceType::CUDA, EncodingStrategyType::HardwareNVENC, {"nvh264enc", "h264parse", "rtph264pay"}});
            GST_DEBUG("[REMOTE] Checking for NVENC strategy support for h264.");
        }
#endif

        if (checkGstreamerPlugins({"vtenc_h264_hw"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("vtenc_h264_hw");
            if (factory) {
                combinations.push_back({DeviceType::CPU, EncodingStrategyType::HardwareVideoToolbox, {"vtenc_h264_hw", "h264parse", "rtph264pay"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[REMOTE] Checking for VideoToolbox strategy support for h264.");
            }
        }

        if (checkGstreamerPlugins({"v4l2h264enc"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("v4l2h264enc");
            if (factory) {
                combinations.push_back({DeviceType::CPU, EncodingStrategyType::HardwareV4L2, {"v4l2h264enc", "h264parse", "rtph264pay"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[REMOTE] Checking for V4L2 strategy support for h264.");
            }
        }

        if (checkGstreamerPlugins({"mfh264enc"}, true) == Result::SUCCESS) {
            GstElementFactory* factory = gst_element_factory_find("mfh264enc");
            if (factory) {
                combinations.push_back({DeviceType::CPU, EncodingStrategyType::HardwareMediaFoundation, {"mfh264enc", "h264parse", "rtph264pay"}});
                gst_object_unref(GST_OBJECT(factory));
                GST_DEBUG("[REMOTE] Checking for MediaFoundation strategy support for h264.");
            }
        }

        combinations.push_back({DeviceType::CPU, EncodingStrategyType::Software, {"openh264enc", "h264parse", "rtph264pay"}});
    }

    if (codec == CodecType::VP8) {
        combinations.push_back({DeviceType::CPU, EncodingStrategyType::Software, {"vp8enc", "rtpvp8pay"}});
    }

    if (codec == CodecType::VP9) {
        combinations.push_back({DeviceType::CPU, EncodingStrategyType::Software, {"vp9enc", "rtpvp9pay"}});
    }

    return combinations;
}

Instance::Remote::EncoderType Instance::Remote::Impl::EncoderFromEncodingStrategy(EncodingStrategyType strategy) {
    switch (strategy) {
        case EncodingStrategyType::Software:
            return EncoderType::Software;
        case EncodingStrategyType::HardwareNVENC:
            return EncoderType::NVENC;
        case EncodingStrategyType::HardwareV4L2:
            return EncoderType::V4L2;
        case EncodingStrategyType::HardwareVideoToolbox:
            return EncoderType::VideoToolbox;
        case EncodingStrategyType::HardwareMediaFoundation:
            return EncoderType::MediaFoundation;
        default:
            return EncoderType::Auto;
    }
}

bool Instance::Remote::Impl::EncoderMatchesStrategy(EncoderType encoder, EncodingStrategyType strategy) {
    return EncoderFromEncodingStrategy(strategy) == encoder;
}

Result Instance::Remote::Impl::createStream() {
    JST_DEBUG("[REMOTE] Creating stream.");

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    gst_init_static_plugins();

    std::vector<std::string> plugins = {
        "appsrc",
        "appsink",
        "capsfilter",
        "clocksync",
        "funnel",
        "nicesink",
        "nicesrc",
        "queue",
        "rawvideoparse",
        "rtpbin",
        "rtpfunnel",
        "rtphdrextmid",
        "rtphdrextrepairedstreamid",
        "rtphdrextstreamid",
        "rtphdrexttwcc",
        "rtpjitterbuffer",
        "rtpptdemux",
        "rtpreddec",
        "rtpredenc",
        "rtprtxreceive",
        "rtprtxsend",
        "rtpsession",
        "rtpssrcdemux",
        "rtpstorage",
        "rtpulpfecdec",
        "rtpulpfecenc",
        "tee",
        "videoconvert",
        "webrtcbin",
    };

    JST_CHECK(checkGstreamerPlugins(plugins));

    if (config.codec == Instance::Remote::CodecType::AV1) {
        JST_ERROR("[REMOTE] AV1 remote encoding is not implemented in the native WebRTC path.");
        return Result::ERROR;
    }

    const auto combinations = encodingCombinations(config.codec);

    bool requestedEncoderFound = false;

    for (const auto& combination : combinations) {
        if (config.encoder != Instance::Remote::EncoderType::Auto) {
            if (!EncoderMatchesStrategy(config.encoder, combination.strategy)) {
                continue;
            }

            requestedEncoderFound = true;
        }

        if (checkGstreamerPlugins(combination.plugins, true) == Result::SUCCESS) {
            inputMemoryDevice_ = combination.device;
            encodingStrategy = combination.strategy;

            JST_INFO("[REMOTE] Using {} encoding with {} memory.", GetEncodingStrategyPrettyName(combination.strategy),
                                                                   GetDevicePrettyName(combination.device));

            return Result::SUCCESS;
        }
        JST_DEBUG("[REMOTE] Failed to find plugins: {}", combination.plugins);
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

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    gst_clear_object(&gstCudaContext);
#endif

    encodingStrategy = EncodingStrategyType::None;
    inputMemoryDevice_ = DeviceType::None;

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::checkGstreamerPlugins(const std::vector<std::string>& plugins,
                                                     const bool& silent) {
    for (const auto& plugin : plugins) {
        GstPlugin* gstPlugin = gst_registry_find_plugin(gst_registry_get(), plugin.c_str());
        if (gstPlugin) {
            gst_object_unref(GST_OBJECT(gstPlugin));
            continue;
        }

        GstElementFactory* factory = gst_element_factory_find(plugin.c_str());
        if (factory) {
            gst_object_unref(GST_OBJECT(factory));
            continue;
        }

        {
            if (!silent) {
                JST_ERROR("[REMOTE] Gstreamer plugin or element '{}' is not available.", plugin);
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

        dx = -dx;
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
        if (pressed && !ctrl && !alt && !meta) {
            const unsigned int cp = firstCodepoint(key);
            if (cp >= 0x20 && cp != 0x7F) {
                ImGui::GetIO().AddInputCharacter(cp);
            }
        }
        return;
    }

    JST_TRACE("[REMOTE] Unknown control (kind='{}').", kind);
}

void Instance::Remote::Impl::onMessageCallback(GstWebRTCDataChannel* self, gchar* data, gpointer user_data) {
    (void)self;

    const std::string payload = data ? data : "";
    if (payload.empty() || payload.size() > 16 * 1024) {
        JST_WARN("[REMOTE] Rejected invalid control message size.");
        return;
    }

    nlohmann::json j;

    try {
        j = nlohmann::json::parse(payload);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Bad control JSON: {}", e.what());
        return;
    }

    if (!j.is_object() || !j.contains("kind") || !j["kind"].is_string()) {
        JST_WARN("[REMOTE] Control message is missing a valid kind.");
        return;
    }
    const std::string kind = j["kind"].get<std::string>();

    auto* that = reinterpret_cast<Instance::Remote::Impl*>(user_data);
    try {
        that->handleInput(kind, j);
    } catch (const std::exception& e) {
        JST_WARN("[REMOTE] Rejected malformed control message: {}", e.what());
    }
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

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if (inputMemoryDevice_ == DeviceType::CUDA) {
        gst_cuda_memory_init_once();
        gst_clear_object(&gstCudaContext);

        const auto& cudaState = Backend::State<DeviceType::CUDA>();
        gstCudaContext = gst_cuda_context_new_wrapped(cudaState->getContext(), cudaState->getDevice());
        if (!gstCudaContext) {
            JST_ERROR("[REMOTE] Failed to wrap CUDA context for GStreamer.");
            gst_object_unref(pipeline);
            pipeline = nullptr;
            return Result::ERROR;
        }
    }
#endif

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

    if (encodingStrategy == EncodingStrategyType::HardwareMediaFoundation) {
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
                elements["encoder"] = encoder = gst_element_factory_make("mfh264enc", "encoder");
                elementOrder.push_back("encoder");

                g_object_set(elements["encoder"], "bitrate", 25*1024, nullptr);

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

    initialFrameTime = std::chrono::steady_clock::now();
    lastKeyframeTime = initialFrameTime;
    forceKeyframe.store(true);
    streaming = true;

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::stopStream() {
    JST_DEBUG("[REMOTE] Stopping stream.");

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
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        gst_clear_object(&gstCudaContext);
#endif
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

void Instance::Remote::Impl::handleSignallerMessage(const std::string& payload) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(payload);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Bad signaller JSON: {}", e.what());
        return;
    }

    if (!j.is_object() || !j.contains("type") || !j["type"].is_string()) {
        JST_ERROR("[REMOTE] Invalid signaller message.");
        return;
    }

    const std::string type = j["type"].get<std::string>();
    if (type == "welcome") {
        if (!j.contains("peerId") || !j["peerId"].is_string()) {
            JST_ERROR("[REMOTE] Invalid signaller welcome message.");
            {
                std::lock_guard<std::mutex> lock(roomMutex);
                roomFailed = true;
            }
            roomCondition.notify_all();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(roomMutex);
            signallerReady = true;
        }
        roomCondition.notify_all();
        return;
    }

    if (type == "roomCreated") {
        if (!j.contains("roomId") || !j["roomId"].is_string() ||
            !j.contains("consumerToken") || !j["consumerToken"].is_string() ||
            !j.contains("clientDomain") || !j["clientDomain"].is_string()) {
            JST_ERROR("[REMOTE] Invalid roomCreated message.");
            std::lock_guard<std::mutex> lock(roomMutex);
            roomFailed = true;
            roomCondition.notify_all();
            return;
        }

        const auto roomId = j["roomId"].get<std::string>();
        const auto token = j["consumerToken"].get<std::string>();
        const auto domain = j["clientDomain"].get<std::string>();
        const bool secureDomain = domain.starts_with("https://");
        const bool insecureDomain = domain.starts_with("http://");
        if (roomId.empty() || roomId.size() > 64 || token.empty() || token.size() > 128 ||
            domain.size() > 2048 || (!secureDomain && !insecureDomain)) {
            JST_ERROR("[REMOTE] Rejected invalid remote room parameters.");
            std::lock_guard<std::mutex> lock(roomMutex);
            roomFailed = true;
            roomCondition.notify_all();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(roomMutex);
            roomId_ = roomId;
            consumerToken = token;
            clientDomain = domain;
            roomReady = true;
        }
        roomCondition.notify_all();
        return;
    }

    if (type == "roomState") {
        if (!j.contains("waiting") || !j["waiting"].is_array() ||
            !j.contains("active") || !j["active"].is_array() ||
            j["waiting"].size() > 256 || j["active"].size() > 256) {
            JST_ERROR("[REMOTE] Invalid roomState message.");
            return;
        }

        std::vector<std::string> waiting;
        std::vector<ClientInfo> active;
        for (const auto& session : j["waiting"]) {
            if (!session.is_string() || session.get_ref<const std::string&>().size() > 64) {
                JST_ERROR("[REMOTE] Invalid waiting session identifier.");
                return;
            }
            waiting.push_back(session.get<std::string>());
        }
        for (const auto& session : j["active"]) {
            if (!session.is_string() || session.get_ref<const std::string&>().size() > 64) {
                JST_ERROR("[REMOTE] Invalid active session identifier.");
                return;
            }
            active.push_back({session.get<std::string>()});
        }

        std::lock_guard<std::mutex> lock(remoteStateMutex);
        waitlist_ = std::move(waiting);
        clients_ = std::move(active);
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
        const std::string details = j.contains("details") && j["details"].is_string()
                                      ? j["details"].get<std::string>()
                                      : "unknown";
        JST_ERROR("[REMOTE] Signaller error: {}", details);
        {
            std::lock_guard<std::mutex> lock(roomMutex);
            if (!roomReady) {
                roomFailed = true;
            }
        }
        roomCondition.notify_all();
        return;
    }

    JST_TRACE("[REMOTE] Ignoring signaller message type '{}'.", type);
}

void Instance::Remote::Impl::handleStartSession(const nlohmann::json& j) {
    if (!j.contains("sessionId") || !j["sessionId"].is_string() ||
        !j.contains("peerId") || !j["peerId"].is_string()) {
        JST_ERROR("[REMOTE] Invalid startSession message.");
        return;
    }

    const std::string sessionId = j["sessionId"].get<std::string>();
    const std::string peerId = j["peerId"].get<std::string>();
    if (sessionId.empty() || sessionId.size() > 64 || peerId.empty() || peerId.size() > 64) {
        JST_ERROR("[REMOTE] Invalid startSession identifiers.");
        return;
    }

    if (createWebRtcSession(sessionId, peerId) != Result::SUCCESS) {
        (void)sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}});
        return;
    }

    JST_INFO("[REMOTE] Starting WebRTC session '{}' with peer '{}'.", sessionId, peerId);

    if (j.contains("offer") && !j["offer"].is_null()) {
        if (!j["offer"].is_string() || j["offer"].get_ref<const std::string&>().size() > 128 * 1024) {
            JST_ERROR("[REMOTE] Invalid startSession offer.");
            destroyWebRtcSession(sessionId);
            (void)sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}});
            return;
        }
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
    if (!j.contains("sessionId") || !j["sessionId"].is_string()) {
        JST_ERROR("[REMOTE] Peer message without a valid sessionId.");
        return;
    }
    const std::string sessionId = j["sessionId"].get<std::string>();
    if (sessionId.empty() || sessionId.size() > 64) {
        JST_ERROR("[REMOTE] Peer message has an invalid sessionId.");
        return;
    }

    if (j.contains("sdp")) {
        const auto& sdp = j["sdp"];
        if (!sdp.is_object() || !sdp.contains("type") || !sdp["type"].is_string() ||
            !sdp.contains("sdp") || !sdp["sdp"].is_string() ||
            sdp["sdp"].get_ref<const std::string&>().size() > 128 * 1024) {
            JST_ERROR("[REMOTE] Peer message has an invalid session description.");
            return;
        }
        if (applyRemoteDescription(sessionId, sdp["type"].get<std::string>(),
                                   sdp["sdp"].get<std::string>()) != Result::SUCCESS) {
            destroyWebRtcSession(sessionId);
            (void)sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}});
        }
        return;
    }

    if (j.contains("ice")) {
        const auto& ice = j["ice"];
        if (!ice.is_object() || !ice.contains("candidate") || !ice["candidate"].is_string() ||
            ice["candidate"].get_ref<const std::string&>().size() > 8 * 1024) {
            JST_ERROR("[REMOTE] Peer message has an invalid ICE candidate.");
            return;
        }

        guint mlineIndex = 0;
        if (ice.contains("sdpMLineIndex") && !ice["sdpMLineIndex"].is_null()) {
            if (!ice["sdpMLineIndex"].is_number_unsigned() ||
                ice["sdpMLineIndex"].get<uint64_t>() > 65535) {
                JST_ERROR("[REMOTE] Peer message has an invalid ICE media index.");
                return;
            }
            mlineIndex = ice["sdpMLineIndex"].get<guint>();
        }

        const std::string candidate = ice["candidate"].get<std::string>();
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
    if (!j.contains("sessionId") || !j["sessionId"].is_string()) {
        JST_ERROR("[REMOTE] Invalid endSession message.");
        return;
    }

    const std::string sessionId = j["sessionId"].get<std::string>();
    if (sessionId.empty() || sessionId.size() > 64) {
        JST_ERROR("[REMOTE] Invalid endSession identifier.");
        return;
    }

    JST_INFO("[REMOTE] WebRTC session '{}' ended.", sessionId);
    destroyWebRtcSession(sessionId);
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

        GstBuffer* buffer = nullptr;

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (inputMemoryDevice_ == DeviceType::CUDA) {
            if (!gstCudaContext) {
                JST_ERROR("[REMOTE] CUDA frame submitted without a GStreamer CUDA context.");
                return Result::ERROR;
            }

            GstVideoInfo info;
            gst_video_info_set_format(&info,
                                      GST_VIDEO_FORMAT_BGRA,
                                      static_cast<guint>(size.x),
                                      static_cast<guint>(size.y));

            GstMemory* memory = gst_cuda_allocator_alloc_wrapped(nullptr,
                                                                 gstCudaContext,
                                                                 nullptr,
                                                                 &info,
                                                                 reinterpret_cast<CUdeviceptr>(const_cast<void*>(data)),
                                                                 this,
                                                                 &OnBufferReleaseCallback);
            if (!memory) {
                JST_ERROR("[REMOTE] Failed to wrap CUDA frame for GStreamer.");
                return Result::ERROR;
            }

            GST_MINI_OBJECT_FLAG_SET(memory, GST_MEMORY_FLAG_READONLY);

            buffer = gst_buffer_new();
            gst_buffer_append_memory(buffer, memory);
            gst_buffer_add_video_meta_full(buffer,
                                           GST_VIDEO_FRAME_FLAG_NONE,
                                           GST_VIDEO_FORMAT_BGRA,
                                           static_cast<guint>(size.x),
                                           static_cast<guint>(size.y),
                                           GST_VIDEO_INFO_N_PLANES(&info),
                                           info.offset,
                                           info.stride);
        }
#endif

        if (!buffer) {
            buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                 const_cast<void*>(data),
                                                 size.x * size.y * 4,
                                                 0,
                                                 size.x * size.y * 4,
                                                 this,
                                                 &OnBufferReleaseCallback);
        }

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
        case EncodingStrategyType::HardwareMediaFoundation:
            return "Hardware Windows (MediaFoundation)";
        default:
            return "Unknown";
    }
}

}  // namespace Jetstream
