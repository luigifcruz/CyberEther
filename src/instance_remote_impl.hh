#pragma once

#include "jetstream/instance_remote.hh"
#include "jetstream/viewport/capture.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include <gst/cuda/gstcuda.h>
#endif

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace Jetstream {

struct Instance::Remote::Impl {
    ~Impl() = default;

    bool supported() const;
    Result create(const Instance::Remote::Config& config);
    Result destroy();
    Result rollbackCreate();
    Result processInput();
    Result captureFrame();
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
    mutable std::mutex remoteStateMutex;
    std::map<CodecType, std::vector<EncoderType>> availableEncoderCache;

    enum class EncodingStrategyType {
        None,
        Software,
        HardwareNVENC,
        HardwareV4L2,
        HardwareVideoToolbox,
        HardwareMediaFoundation,
    };

    struct EncodingCombination {
        DeviceType device = DeviceType::None;
        EncodingStrategyType strategy = EncodingStrategyType::None;
        std::vector<std::string> plugins;
    };

    Extent2D<U64> size;

    DeviceType inputMemoryDevice_ = DeviceType::None;
    EncodingStrategyType encodingStrategy = EncodingStrategyType::None;

    // Broker state
    std::string clientDomain;
    std::string signallerUrl;
    std::mutex roomMutex;
    std::condition_variable roomCondition;
    bool signallerReady = false;
    bool roomReady = false;
    bool roomFailed = false;

    Result createBroker();
    Result destroyBroker();
    Result createRoom();

    // Stream
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* encoder = nullptr;
    GstElement* tee = nullptr;

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    GstCudaContext* gstCudaContext = nullptr;
#endif

    std::unique_ptr<httplib::ws::WebSocketClient> signallerClient;
    std::thread signallerThread;
    std::atomic<bool> signallerRunning = false;
    std::atomic<socket_t> signallerSocket = INVALID_SOCKET;
    std::mutex signallerMutex;

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

    struct ControlChannelContext {
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
    std::vector<EncoderType> available(CodecType codec);
    std::vector<EncodingCombination> encodingCombinations(CodecType codec);
    static EncoderType EncoderFromEncodingStrategy(EncodingStrategyType strategy);
    static bool EncoderMatchesStrategy(EncoderType encoder, EncodingStrategyType strategy);
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
    struct QueuedInput {
        std::string sessionId;
        nlohmann::json payload;
        bool reset = false;
    };

    struct RemoteInputState {
        std::unordered_set<ImGuiKey> keys;
        bool alt = false;
        bool ctrl = false;
        bool shift = false;
        bool meta = false;
    };

    std::mutex inputMutex;
    std::deque<QueuedInput> inputQueue;
    std::unordered_map<std::string, RemoteInputState> remoteInputStates;
    std::unordered_set<ImGuiKey> appliedRemoteKeys;

    void enqueueInput(std::string sessionId, nlohmann::json payload);
    void enqueueInputReset(const std::string& sessionId);
    void handleInput(const std::string& sessionId, const std::string& kind, const nlohmann::json& j);
    void resetInput(const std::string& sessionId);
    void synchronizeKeyboardState();
    void createControlChannel(const std::string& sessionId);
    static void onMessageCallback(GstWebRTCDataChannel* self, gchar* data, gpointer user_data);
    static void onChannelClosedCallback(GstWebRTCDataChannel* self, gpointer user_data);
    static void onChannelCallback(GstElement* self, GstWebRTCDataChannel* channel, gpointer user_data);
    static void onIceCandidateCallback(GstElement* self, guint mlineIndex, gchar* candidate, gpointer user_data);
    static void onOfferCreatedCallback(GstPromise* promise, gpointer user_data);
    static void onAnswerCreatedCallback(GstPromise* promise, gpointer user_data);

    struct WebRtcPromiseContext {
        Impl* impl = nullptr;
        std::string sessionId;
    };

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

}  // namespace Jetstream
