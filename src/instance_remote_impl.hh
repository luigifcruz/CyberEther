#pragma once

#include "instance_remote_broker.hh"
#include "instance_remote_encoder.hh"

#include "jetstream/instance_remote.hh"
#include "jetstream/viewport/capture.hh"
#include "jetstream/render/base/input.hh"
#include "jetstream/surface.hh"

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

#include <rtc/rtc.hpp>

#include <httplib.h>
#ifdef ERROR
#undef ERROR
#endif
#include <nlohmann/json.hpp>

namespace Jetstream {

struct Instance::Remote::Impl {
    ~Impl() = default;

    bool supported() const;
    Result create(const Instance::Remote::Config& config);
    Result destroy();
    Result rollbackCreate();
    Result processInput();
    void requestKeyframe();
    Result captureFrame(bool presented,
                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());
    Result approveClient(const std::string& sessionId);
    Result removeClient(const std::string& sessionId);

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

    struct EncodingCombination {
        DeviceType device = DeviceType::None;
        EncoderType encoder = EncoderType::Auto;
    };

    Extent2D<U64> size;

    DeviceType inputMemoryDevice_ = DeviceType::None;
    EncoderType encoderType = EncoderType::Auto;
    std::vector<EncodingCombination> encoderFallbacks;

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
    std::unique_ptr<Encoder> encoder;
    std::vector<Encoder::Packet> packets;

    std::unique_ptr<Broker> broker;
    std::unique_ptr<httplib::ws::WebSocketClient> signallerClient;
    std::thread signallerThread;
    std::atomic<bool> signallerRunning = false;
    socket_t signallerSocket = INVALID_SOCKET;
    std::mutex signallerMutex;

    struct WebRtcSession {
        struct Keyframe {
            std::mutex mutex;
            Impl* impl = nullptr;
        };

        std::string sessionId;
        std::string peerId;
        std::shared_ptr<rtc::PeerConnection> peer;
        std::shared_ptr<rtc::Track> track;
        std::shared_ptr<rtc::DataChannel> control;
        std::shared_ptr<Keyframe> keyframe;

        std::mutex candidatesMutex;
        bool described = false;
        U64 received = 0;
        std::vector<std::pair<std::string, std::string>> candidates;
    };

    std::mutex sessionsMutex;
    std::unordered_map<std::string, std::shared_ptr<WebRtcSession>> sessions;

    std::atomic<bool> streaming = false;
    std::mutex streamMutex;

    Result createStream();
    Result startStream();
    Result stopStream();
    Result destroyStream();
    std::vector<EncoderType> available(CodecType codec);
    std::vector<EncodingCombination> encodingCombinations(CodecType codec);
    Result createWebRtcSession(const std::string& sessionId, const std::string& peerId);
    void destroyWebRtcSession(const std::string& sessionId);
    void destroyAllWebRtcSessions();
    std::shared_ptr<WebRtcSession> findSession(const std::string& sessionId);
    Result startSignaller();
    Result stopSignaller();
    httplib::ws::Result connectSignaller(const std::string& url);
    void closeSignallerClient();
    void signallerLoop();
    void readSignaller();
    void handleSignallerMessage(const std::string& payload);
    void handleStartSession(const nlohmann::json& j);
    void handlePeerMessage(const nlohmann::json& j);
    void handleEndSession(const nlohmann::json& j);
    bool sendSignallerMessage(const nlohmann::json& j);
    void sendSessionDescription(const std::string& sessionId, const rtc::Description& description);
    void sendIceCandidate(const std::string& sessionId, const rtc::Candidate& candidate);
    void addRemoteCandidate(WebRtcSession& session, const std::string& candidate, const std::string& mid);
    Result createVideoTrack(WebRtcSession& session);
    void closeWebRtcSession(WebRtcSession& session);
    Result applyRemoteDescription(const std::string& sessionId,
                                  const std::string& type,
                                  const std::string& sdp);
    struct QueuedInput {
        std::string sessionId;
        nlohmann::json payload;
        bool reset = false;
    };

    struct RemoteInputState {
        std::unordered_set<KeyCode> keys;
        std::unordered_set<MouseButton> mouseButtons;
        bool alt = false;
        bool ctrl = false;
        bool shift = false;
        bool meta = false;
    };

    std::mutex inputMutex;
    std::deque<std::string> inputSessionOrder;
    std::deque<QueuedInput> inputQueue;
    std::unordered_map<std::string, RemoteInputState> remoteInputStates;
    std::unordered_set<KeyCode> appliedRemoteKeys;

    void enqueueInput(std::string sessionId, nlohmann::json payload);
    void enqueueInputReset(const std::string& sessionId);
    void handleInput(const std::string& sessionId, const std::string& kind, const nlohmann::json& j);
    void resetInput(const std::string& sessionId);
    void synchronizeKeyboardState();
    void pushInput(Render::InputQueue::Event event);
    Extent2D<F32> framebufferScale() const;
    void bindControlChannel(const std::string& sessionId, const std::shared_ptr<rtc::DataChannel>& channel);
    void handleControlMessage(const std::string& sessionId, std::string_view payload);

    std::unique_ptr<Viewport::FrameCapture> frameCapture;
    std::chrono::steady_clock::time_point nextCaptureTime{};
    bool capturePending = false;
    std::atomic<bool> captureRequested = false;
    std::thread frameSubmissionThread;
    std::atomic<bool> frameSubmissionRunning = false;

    // Frame submission
    std::atomic<bool> forceKeyframe = false;
    std::chrono::time_point<std::chrono::steady_clock> initialFrameTime;
    std::chrono::time_point<std::chrono::steady_clock> lastKeyframeTime;

    Result pushNewFrame(const void* data);
};

}  // namespace Jetstream
