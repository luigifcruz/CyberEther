#include "instance_remote_impl.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/capture.hh"
#include "jetstream/viewport/adapters/generic.hh"
#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/viewport/adapters/vulkan.hh"
#include "jetstream/viewport/capture/vulkan.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/viewport/adapters/metal.hh"
#include "jetstream/viewport/capture/metal.hh"
#endif

#include <algorithm>
#include <cmath>
#include <cctype>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <map>
#include <random>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#ifdef JST_OS_WINDOWS
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#endif

namespace Jetstream {

namespace {

bool IsIpAddress(const std::string& address) {
    in6_addr buffer = {};
    return inet_pton(AF_INET, address.c_str(), &buffer) == 1 ||
           inet_pton(AF_INET6, address.c_str(), &buffer) == 1;
}

constexpr const char* kVideoMid = "video";
constexpr const char* kCname = "cyberether";
constexpr U8 kPayloadType = 96;
constexpr int kPlayoutDelayExtension = 5;
constexpr const char* kPlayoutDelayUri = "http://www.webrtc.org/experiments/rtp-hdrext/playout-delay";
constexpr U16 kPlayoutDelayMaximum = 2;
constexpr U64 kMaximumFragmentSize = 1100;
constexpr U64 kMaximumCandidates = 64;
constexpr U64 kNackHistory = 2048;
constexpr U64 kMaxClipboardBytes = 1024 * 1024;
constexpr U64 kClipboardChunkBytes = 8 * 1024;
constexpr U64 kMaxClipboardChunks = 256;
constexpr U64 kMaxControlBytes = 16 * 1024;
constexpr U64 kMaxClipboardMessageBytes = 64 * 1024;
constexpr U64 kTelemetryLimit = 4096;

std::vector<std::string> SplitUtf8(const std::string& text, const U64 limit) {
    std::vector<std::string> chunks;
    U64 start = 0;
    while (start < text.size()) {
        U64 end = std::min<U64>(start + limit, text.size());
        while (end < text.size() && end > start && (static_cast<U8>(text[end]) & 0xC0) == 0x80) {
            --end;
        }
        if (end == start) {
            end = std::min<U64>(start + limit, text.size());
        }
        chunks.push_back(text.substr(start, end - start));
        start = end;
    }
    if (chunks.empty()) {
        chunks.emplace_back();
    }
    return chunks;
}

}  // namespace

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
    if (!IsRemotePortRangeValid(config.media.minPort, config.media.maxPort)) {
        JST_ERROR("[REMOTE] Invalid port range {}-{}.", config.media.minPort, config.media.maxPort);
        return Result::ERROR;
    }
    if (!config.media.address.empty() && !IsIpAddress(config.media.address)) {
        JST_ERROR("[REMOTE] Invalid media address '{}'. Expected an IP address.", config.media.address);
        return Result::ERROR;
    }
    return impl->create(config);
}

Result Instance::Remote::destroy() {
    return impl->destroy();
}

bool Instance::Remote::started() const {
    return impl->started_;
}

Result Instance::Remote::captureFrame(bool presented) {
    return impl->captureFrame(presented);
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

Instance::Remote::Stats Instance::Remote::stats() const {
    return impl->stats();
}

Result Instance::Remote::approveClient(const std::string& sessionId) {
    return impl->approveClient(sessionId);
}

Result Instance::Remote::removeClient(const std::string& sessionId) {
    return impl->removeClient(sessionId);
}

bool Instance::Remote::Impl::supported() const {
    switch (this->viewportDevice) {
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        case DeviceType::Vulkan:
            return true;
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        case DeviceType::Metal:
            return true;
#endif
        default:
            return false;
    }
}

Result Instance::Remote::Impl::create(const Instance::Remote::Config& config) {
    JST_DEBUG("[REMOTE] Initializing remote streaming.");

    if (this->started_) {
        JST_ERROR("[REMOTE] Remote streaming is already started.");
        return Result::ERROR;
    }

    // A failed attempt must leave the same clean state as a fresh instance.
    if (this->rollbackCreate() != Result::SUCCESS) {
        JST_ERROR("[REMOTE] Failed to clean stale remote resources before creation.");
        return Result::ERROR;
    }

    this->config = config;

    if (config.video.framerate == 0) {
        JST_ERROR("[REMOTE] Frame rate must be greater than zero.");
        return Result::ERROR;
    }

    if (!supported()) {
        JST_ERROR("[REMOTE] Current backend ({}) is not supported for remote streaming.", this->viewportDevice);
        return Result::ERROR;
    }

    this->setStreamSize(this->viewportExtent());
    if (viewport && (this->launchSize.x == 0 || this->launchSize.y == 0)) {
        this->launchSize = this->size;
        const auto display = viewport->displaySize();
        const auto framebuffer = viewport->framebufferSize();
        this->launchScale = display.x > 0.0f ? framebuffer.x / display.x : 1.0f;
    }

    if (this->size.x == 0 || this->size.y == 0) {
        JST_ERROR("[REMOTE] Failed to get viewport size.");
        return Result::ERROR;
    }

    if (this->createStream() != Result::SUCCESS) {
        this->rollbackCreate();
        return Result::ERROR;
    }
    if (this->createCapture() != Result::SUCCESS) {
        if (this->inputMemoryDevice_ == DeviceType::CPU) {
            this->rollbackCreate();
            return Result::ERROR;
        }

        JST_WARN("[REMOTE] Failed to share frames with the encoder. Falling back to CPU memory.");
        if (this->createStream(DeviceType::CPU) != Result::SUCCESS || this->createCapture() != Result::SUCCESS) {
            this->rollbackCreate();
            return Result::ERROR;
        }
    }

    if (this->createBroker() != Result::SUCCESS) {
        this->rollbackCreate();
        return Result::ERROR;
    }

    if (this->startSubmission() != Result::SUCCESS) {
        this->rollbackCreate();
        return Result::ERROR;
    }

    if (viewport) {
        std::lock_guard<std::mutex> lock(inputMutex);
        sentClipboardRevision = viewport->clipboardRevision();
    }

    this->started_ = true;
    JST_INFO("[REMOTE] Remote streaming started.");
    return Result::SUCCESS;
}

Extent2D<U64> Instance::Remote::Impl::viewportExtent() const {
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (this->viewportDevice == DeviceType::Metal) {
        if (auto* adapter = dynamic_cast<Viewport::Adapter<DeviceType::Metal>*>(this->viewport)) {
            return adapter->getSwapchainExtent();
        }
    }
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (this->viewportDevice == DeviceType::Vulkan) {
        if (auto* adapter = dynamic_cast<Viewport::Adapter<DeviceType::Vulkan>*>(this->viewport)) {
            return adapter->getSwapchainExtent();
        }
    }
#endif
    return {};
}

Result Instance::Remote::Impl::startSubmission() {
    this->frameSubmissionRunning = true;
    const bool shared = this->inputMemoryDevice_ == DeviceType::Vulkan;
    try {
        this->frameSubmissionThread = std::thread([this, shared]() {
            const auto push = [this](const void* data) {
                const Result pushed = this->pushNewFrame(data);
                this->releaseCapture();
                if (pushed != Result::SUCCESS) {
                    this->requestKeyframe();
                }
            };

            while (this->frameSubmissionRunning) {
                if (shared) {
                    Viewport::FrameCapture::SharedFrame frame;
                    if (this->frameCapture->getSharedFrame(frame) == Result::SUCCESS) {
                        push(&frame);
                    }
                    continue;
                }

                Tensor tensor;
                if (this->frameCapture->getFrameData(tensor) == Result::SUCCESS) {
                    push(tensor.data());
                }
            }
        });
    } catch (const std::exception& e) {
        this->frameSubmissionRunning = false;
        JST_ERROR("[REMOTE] Failed to start frame submission thread: {}", e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

void Instance::Remote::Impl::stopSubmission() {
    this->frameSubmissionRunning = false;
    if (this->frameCapture) {
        (void)this->frameCapture->stop();
    }
    if (this->frameSubmissionThread.joinable()) {
        this->frameSubmissionThread.join();
    }
}

Result Instance::Remote::Impl::followViewport() {
    const Extent2D<U64> current = this->viewportExtent();
    if (current.x == 0 || current.y == 0 || (current.x == this->size.x && current.y == this->size.y)) {
        return Result::SUCCESS;
    }
    return this->rebuildStream(current);
}

Result Instance::Remote::Impl::rebuildStream(const Extent2D<U64>& next) {
    JST_INFO("[REMOTE] Viewport changed from {}x{} to {}x{}. Rebuilding the stream.",
             this->size.x, this->size.y, next.x, next.y);

    std::lock_guard<std::mutex> lifecycle(lifecycleMutex);

    this->stopSubmission();

    std::unique_ptr<Encoder> oldEncoder;
    {
        std::lock_guard<std::mutex> lock(streamMutex);
        oldEncoder = std::move(encoder);
    }
    oldEncoder.reset();

    this->setStreamSize(next);
    this->nextCaptureTime = {};
    this->capturePending = false;
    this->captureBlocked = false;

    if (this->inputMemoryDevice_ == DeviceType::None && this->createStream() != Result::SUCCESS) {
        JST_ERROR("[REMOTE] No encoder supports {}x{}.", next.x, next.y);
        return Result::ERROR;
    }

    if (this->createCapture() != Result::SUCCESS) {
        JST_WARN("[REMOTE] Failed to share frames with the encoder. Falling back to CPU memory.");
        if (this->inputMemoryDevice_ == DeviceType::CPU ||
            this->createStream(DeviceType::CPU) != Result::SUCCESS ||
            this->createCapture() != Result::SUCCESS) {
            JST_ERROR("[REMOTE] Failed to rebuild the frame capture.");
            return Result::ERROR;
        }
    }

    JST_CHECK(this->startSubmission());

    if (!this->streaming) {
        return Result::SUCCESS;
    }

    if (this->reopenStream() == Result::SUCCESS) {
        return Result::SUCCESS;
    }

    JST_WARN("[REMOTE] Encoder can't stream at {}x{}. Picking another encoder.", next.x, next.y);
    this->stopSubmission();
    if (this->createStream() != Result::SUCCESS || this->createCapture() != Result::SUCCESS) {
        if (this->createStream(DeviceType::CPU) != Result::SUCCESS || this->createCapture() != Result::SUCCESS) {
            JST_ERROR("[REMOTE] Failed to rebuild the frame capture.");
            return Result::ERROR;
        }
    }
    JST_CHECK(this->startSubmission());
    return this->reopenStream();
}

Result Instance::Remote::Impl::reopenStream() {
    std::unique_ptr<Encoder> newEncoder;
    JST_CHECK(this->openEncoder(newEncoder));
    {
        std::lock_guard<std::mutex> lock(telemetry.mutex);
        telemetry.samples.clear();
        telemetry.format = newEncoder->format();
    }
    {
        std::lock_guard<std::mutex> lock(streamMutex);
        encoder = std::move(newEncoder);
    }
    requestKeyframe();
    return Result::SUCCESS;
}

Extent2D<U64> Instance::Remote::Impl::streamSize() const {
    std::lock_guard<std::mutex> lock(sizeMutex);
    return this->size;
}

void Instance::Remote::Impl::setStreamSize(const Extent2D<U64>& next) {
    std::lock_guard<std::mutex> lock(sizeMutex);
    this->size = next;
}

void Instance::Remote::Impl::restoreViewport() {
    bool resized = false;
    {
        std::lock_guard<std::mutex> lock(inputMutex);
        resized = appliedViewport != ViewportRequest{};
        viewportRequests.clear();
        appliedViewport = {};
    }

    if (resized && viewport && launchSize.x > 0 && launchSize.y > 0) {
        (void)viewport->resize(launchSize, launchScale);
    }
}

Result Instance::Remote::Impl::createCapture() {
    if (this->frameCapture) {
        (void)this->frameCapture->destroy();
        this->frameCapture.reset();
    }

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    if (this->viewportDevice == DeviceType::Metal) {
        this->frameCapture = std::make_unique<Viewport::FrameCaptureMetal>([this](bool available) {
            return this->acceptCapture(available, std::chrono::steady_clock::now());
        });
    }
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (this->viewportDevice == DeviceType::Vulkan) {
        const auto exporting = this->inputMemoryDevice_ == DeviceType::Vulkan ? Encoder::ExportFor(this->encoderType)
                                                                              : Encoder::Export::None;
        this->frameCapture = std::make_unique<Viewport::FrameCaptureVulkan>(exporting);
    }
#endif

    if (!this->frameCapture) {
        JST_ERROR("[REMOTE] No frame capture available for device type.");
        return Result::ERROR;
    }

    if (this->frameCapture->create(this->viewport, this->inputMemoryDevice_) != Result::SUCCESS) {
        (void)this->frameCapture->destroy();
        this->frameCapture.reset();
        return Result::ERROR;
    }

    this->setStreamSize(this->frameCapture->extent());
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::destroy() {
    JST_DEBUG("[REMOTE] Destroying remote streaming.");

    const Result result = this->rollbackCreate();

    JST_INFO("[REMOTE] Remote streaming stopped.");
    return result;
}

Result Instance::Remote::Impl::rollbackCreate() {
    Result result = Result::SUCCESS;
    this->started_ = false;
    this->nextCaptureTime = {};
    this->capturePending = false;
    this->captureBlocked = false;

    {
        std::lock_guard<std::mutex> lock(inputMutex);
        inputQueue.clear();
    }
    this->restoreViewport();
    while (!remoteInputStates.empty()) {
        resetInput(remoteInputStates.begin()->first);
    }
    if (!appliedRemoteKeys.empty()) {
        synchronizeKeyboardState();
    }

    this->frameSubmissionRunning = false;
    if (this->frameCapture && this->frameCapture->stop() != Result::SUCCESS) {
        result = Result::ERROR;
    }

    // Stream teardown releases any frame still retained by the encoder, which
    // wakes the submission thread before it is joined below.
    if (this->destroyBroker() != Result::SUCCESS) {
        result = Result::ERROR;
    }

    if (this->frameSubmissionThread.joinable()) {
        this->frameSubmissionThread.join();
    }

    if (this->frameCapture) {
        if (this->frameCapture->destroy() != Result::SUCCESS) {
            result = Result::ERROR;
        }
        this->frameCapture.reset();
    }

    if (this->destroyStream() != Result::SUCCESS) {
        result = Result::ERROR;
    }

    return result;
}

Result Instance::Remote::Impl::captureFrame(bool presented, std::chrono::steady_clock::time_point now) {
    if (this->started_ && !this->signallerRunning) {
        JST_WARN("[REMOTE] Stopping remote streaming after the signaller disconnected.");
        return this->destroy();
    }

    syncClipboard();
    JST_CHECK(processInput());
    applyViewport();
    if (this->started_ && followViewport() != Result::SUCCESS) {
        JST_WARN("[REMOTE] Video is paused until the viewport returns to a size the encoder supports.");
    }
    if (!this->frameCapture) {
        return Result::SUCCESS;
    }
    if (this->captureRequested.exchange(false)) {
        this->capturePending = true;
    }
    if (this->viewportDevice == DeviceType::Metal) {
        if (this->capturePending && now >= this->nextCaptureTime && !this->captureBlocked && viewport) {
            viewport->requestRedraw();
        }
        return Result::SUCCESS;
    }
    if (now < this->nextCaptureTime) {
        this->capturePending = this->capturePending || presented;
        return Result::SUCCESS;
    }
    if (!presented) {
        if (this->capturePending && !this->captureBlocked && viewport) {
            viewport->requestRedraw();
        }
        return Result::SUCCESS;
    }
    this->captureBlocked = true;
    if (this->frameCapture->captureFrame() == Result::SKIP) {
        this->capturePending = true;
        return Result::SUCCESS;
    }
    this->captureBlocked = false;
    this->scheduleCapture(now);
    return Result::SUCCESS;
}

bool Instance::Remote::Impl::acceptCapture(bool available, std::chrono::steady_clock::time_point now) {
    if (!available) {
        this->captureBlocked = true;
        this->capturePending = true;
        return false;
    }
    if (now < this->nextCaptureTime) {
        this->capturePending = true;
        return false;
    }
    this->captureRequested = false;
    this->scheduleCapture(now);
    return true;
}

void Instance::Remote::Impl::releaseCapture() {
    this->frameCapture->releaseFrame();
    if (this->captureBlocked.exchange(false) && viewport) {
        viewport->requestRedraw();
    }
}

void Instance::Remote::Impl::scheduleCapture(std::chrono::steady_clock::time_point now) {
    const auto interval = std::chrono::nanoseconds(std::chrono::seconds(1)) / config.video.framerate;
    if (now - this->nextCaptureTime >= interval) {
        this->nextCaptureTime = now;
    }
    this->nextCaptureTime += interval;
    this->capturePending = false;
}

void Instance::Remote::Impl::requestKeyframe() {
    forceKeyframe.store(true);
    captureRequested.store(true);
    if (viewport) {
        viewport->requestRedraw();
    }
}

Result Instance::Remote::Impl::processInput() {
    std::deque<QueuedInput> pending;
    {
        std::lock_guard<std::mutex> lock(inputMutex);
        pending.swap(inputQueue);
    }

    ++inputFrame;
    for (auto it = pending.begin(); it != pending.end(); ++it) {
        try {
            if (it->reset) {
                resetInput(it->sessionId);
                lastInputFrame = inputFrame;
                continue;
            }
            const std::string kind = it->payload.value("kind", "");
            if (kind == "clipboard" && lastInputFrame != 0 && inputFrame - lastInputFrame < 2) {
                std::lock_guard<std::mutex> lock(inputMutex);
                inputQueue.insert(inputQueue.begin(), std::make_move_iterator(it), std::make_move_iterator(pending.end()));
                break;
            }
            handleInput(it->sessionId, kind, it->payload);
            if (kind == "keyboard" || kind == "text" || (kind == "mouse" && it->payload.value("act", "") != "move")) {
                lastInputFrame = inputFrame;
            }
        } catch (const std::exception& e) {
            JST_WARN("[REMOTE] Rejected malformed control message: {}", e.what());
        }
    }
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::approveClient(const std::string& sessionId) {
    if (!this->started_) {
        JST_ERROR("[REMOTE] Can't approve client when session is not started.");
        return Result::ERROR;
    }

    bool waiting = false;
    {
        std::lock_guard<std::mutex> lock(remoteStateMutex);
        waiting = !sessionId.empty() && std::ranges::find(waitlist_, sessionId) != waitlist_.end();
    }
    if (!waiting) {
        JST_ERROR("[REMOTE] Client session '{}' not found.", sessionId);
        return Result::ERROR;
    }

    if (!sendSignallerMessage({{"type", "approveSession"}, {"sessionId", sessionId}})) {
        JST_ERROR("[REMOTE] Failed to send client approval.");
        return Result::ERROR;
    }

    JST_INFO("[REMOTE] Client session '{}' approved.", sessionId);
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::removeClient(const std::string& sessionId) {
    if (!this->started_) {
        JST_ERROR("[REMOTE] Can't remove client when session is not started.");
        return Result::ERROR;
    }

    bool known = false;
    {
        std::lock_guard<std::mutex> lock(remoteStateMutex);
        known = !sessionId.empty() &&
                (std::ranges::find(waitlist_, sessionId) != waitlist_.end() ||
                 std::ranges::any_of(clients_, [&](const auto& client) {
                     return client.sessionId == sessionId;
                 }));
    }
    if (!known) {
        JST_ERROR("[REMOTE] Client session '{}' not found.", sessionId);
        return Result::ERROR;
    }

    if (!sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}})) {
        JST_ERROR("[REMOTE] Failed to remove client.");
        return Result::ERROR;
    }

    JST_INFO("[REMOTE] Client session '{}' removed.", sessionId);
    return Result::SUCCESS;
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

    std::vector<EncoderType> encoders;
    for (const auto& combination : encodingCombinations(codec)) {
        if (std::ranges::find(encoders, combination.encoder) == encoders.end()) {
            encoders.push_back(combination.encoder);
        }
    }

    if (!encoders.empty()) {
        encoders.insert(encoders.begin(), EncoderType::Auto);
    }

    availableEncoderCache[codec] = encoders;
    return encoders;
}

std::vector<Instance::Remote::Impl::EncodingCombination> Instance::Remote::Impl::encodingCombinations(CodecType codec,
                                                                                                    DeviceType only) {
    std::vector<EncodingCombination> combinations;

    std::vector<EncodingCombination> copied;

    const auto probe = [&](DeviceType device, EncoderType encoder) {
        Encoder::Config config = {
            .codec = codec,
            .encoder = encoder,
            .device = device,
            .size = streamSize(),
        };
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        if (device == DeviceType::Vulkan) {
            config.adapter = Viewport::FrameCaptureVulkan::Device();
        }
#endif
        return Encoder::Available(config);
    };

    const auto add = [&](EncoderType encoder) {
        bool exported = false;
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        const auto exporting = Encoder::ExportFor(encoder);
        exported = only != DeviceType::CPU &&
                   viewportDevice == DeviceType::Vulkan &&
                   exporting != Encoder::Export::None &&
                   Viewport::FrameCaptureVulkan::SupportsExport(exporting);
#endif
        if (exported && probe(DeviceType::Vulkan, encoder)) {
            combinations.push_back({DeviceType::Vulkan, encoder});
        } else if (probe(DeviceType::CPU, encoder)) {
            copied.push_back({DeviceType::CPU, encoder});
        }
    };

    add(EncoderType::NVENC);
    add(EncoderType::AMF);
    add(EncoderType::QSV);
    add(EncoderType::VAAPI);
    add(EncoderType::VideoToolbox);
    add(EncoderType::V4L2);
    add(EncoderType::Software);

    combinations.insert(combinations.end(), copied.begin(), copied.end());
    return combinations;
}

Result Instance::Remote::Impl::createStream(DeviceType only) {
    JST_DEBUG("[REMOTE] Creating stream.");

    encoderType = EncoderType::Auto;
    inputMemoryDevice_ = DeviceType::None;
    encoderFallbacks.clear();

    for (const auto& combination : encodingCombinations(config.video.codec, only)) {
        if (config.video.encoder != EncoderType::Auto && config.video.encoder != combination.encoder) {
            continue;
        }

        if (encoderType != EncoderType::Auto) {
            const bool sameInput = combination.device == inputMemoryDevice_ &&
                                   (combination.device != DeviceType::Vulkan ||
                                    Encoder::ExportFor(combination.encoder) == Encoder::ExportFor(encoderType));
            if (sameInput) {
                encoderFallbacks.push_back(combination);
            }
            continue;
        }

        inputMemoryDevice_ = combination.device;
        encoderType = combination.encoder;

        JST_INFO("[REMOTE] Using {} encoding with {} memory.", combination.encoder,
                                                               GetDevicePrettyName(combination.device));
    }

    if (encoderType != EncoderType::Auto) {
        return Result::SUCCESS;
    }

    const std::string codecName = Jetstream::GetRemoteCodecName(config.video.codec);

    if (config.video.encoder != EncoderType::Auto) {
        JST_ERROR("[REMOTE] Requested encoder '{}' is not available for codec '{}' on this system.",
                  Jetstream::GetRemoteEncoderPrettyName(config.video.encoder),
                  codecName);
        return Result::ERROR;
    }

    JST_ERROR("[REMOTE] No encoder is available for codec '{}' on this system.", codecName);
    return Result::ERROR;
}

Result Instance::Remote::Impl::destroyStream() {
    JST_DEBUG("[REMOTE] Destroying stream.");

    encoderType = EncoderType::Auto;
    inputMemoryDevice_ = DeviceType::None;
    encoderFallbacks.clear();

    return Result::SUCCESS;
}

void Instance::Remote::Impl::handleInput(const std::string& sessionId,
                                         const std::string& kind,
                                         const nlohmann::json& j) {
    std::lock_guard<std::mutex> lock(inputMutex);
    if (inputSessionOrder.empty() || inputSessionOrder.front() != sessionId) {
        return;
    }

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

    auto mapKey = [](const std::string& code, const std::string& key) -> KeyCode {
        if (code.size() == 4 && code.rfind("Key", 0) == 0) {
            char c = static_cast<char>(std::toupper(static_cast<unsigned char>(code[3])));
            if (c >= 'A' && c <= 'Z') return static_cast<KeyCode>(static_cast<U16>(KeyCode::A) + (c - 'A'));
        }
        if (code.rfind("Digit", 0) == 0 && code.size() == 6 && std::isdigit(static_cast<unsigned char>(code[5]))) {
            char d = code[5];
            return static_cast<KeyCode>(static_cast<U16>(KeyCode::Digit0) + (d - '0'));
        }
        if (!code.empty() && code[0] == 'F' && code.size() <= 3 && std::isdigit(static_cast<unsigned char>(code[1]))) {
            int f = std::clamp(std::stoi(code.substr(1)), 1, 24);
            return static_cast<KeyCode>(static_cast<U16>(KeyCode::F1) + (f - 1));
        }
        if (code == "ArrowLeft")                       return KeyCode::LeftArrow;
        if (code == "ArrowRight")                      return KeyCode::RightArrow;
        if (code == "ArrowUp")                         return KeyCode::UpArrow;
        if (code == "ArrowDown")                       return KeyCode::DownArrow;
        if (code == "AltLeft")                         return KeyCode::LeftAlt;
        if (code == "AltRight")                        return KeyCode::RightAlt;
        if (code == "ControlLeft")                     return KeyCode::LeftCtrl;
        if (code == "ControlRight")                    return KeyCode::RightCtrl;
        if (code == "ShiftLeft")                       return KeyCode::LeftShift;
        if (code == "ShiftRight")                      return KeyCode::RightShift;
        if (code == "MetaLeft")                        return KeyCode::LeftSuper;
        if (code == "MetaRight")                       return KeyCode::RightSuper;
        if (code == "Enter" || key == "Enter")         return KeyCode::Enter;
        if (code == "Escape" || key == "Escape")       return KeyCode::Escape;
        if (code == "Backspace" || key == "Backspace") return KeyCode::Backspace;
        if (code == "Tab" || key == "Tab")             return KeyCode::Tab;
        if (code == "Space" || key == " ")             return KeyCode::Space;
        if (code == "Delete")                          return KeyCode::Delete;
        if (code == "Insert")                          return KeyCode::Insert;
        if (code == "Home")                            return KeyCode::Home;
        if (code == "End")                             return KeyCode::End;
        if (code == "PageUp")                          return KeyCode::PageUp;
        if (code == "PageDown")                        return KeyCode::PageDown;
        if (code == "Minus")                           return KeyCode::Minus;
        if (code == "Equal")                           return KeyCode::Equal;
        if (code == "BracketLeft")                     return KeyCode::LeftBracket;
        if (code == "BracketRight")                    return KeyCode::RightBracket;
        if (code == "Backslash")                       return KeyCode::Backslash;
        if (code == "Semicolon")                       return KeyCode::Semicolon;
        if (code == "Quote")                           return KeyCode::Apostrophe;
        if (code == "Comma")                           return KeyCode::Comma;
        if (code == "Period")                          return KeyCode::Period;
        if (code == "Slash")                           return KeyCode::Slash;
        if (code == "Backquote" ||
            code == "Backtick"  ||
            code == "QuoteLeft")                       return KeyCode::GraveAccent;
        return KeyCode::Unknown;
    };

    if (kind == "mouse") {
        const std::string act = j.value("act", "");
        const double nx = j.value("x", 0.0);
        const double ny = j.value("y", 0.0);

        const int w = size.x;
        const int h = size.y;

        const auto scale = framebufferScale();
        const float px = denorm(nx, w) / scale.x;
        const float py = denorm(ny, h) / scale.y;

        if (act == "move") {
            JST_TRACE("[REMOTE] Mouse: move (x='{}', y='{}', nx='{}', ny='{}')", px, py, nx, ny);
            pushInput(Render::InputQueue::Pointer{Extent2D<F32>{px, py}});
            return;
        }

        if (act == "down" || act == "up" || act == "click" || act == "dblclick") {
            const int domButton = j.value("button", 0);
            const auto pressed = Render::InputQueue::ButtonFromIndex(mapMouseButton(domButton));
            if (!pressed) {
                return;
            }
            const MouseButton b = *pressed;

            pushInput(Render::InputQueue::Pointer{Extent2D<F32>{px, py}});

            if (act == "down") {
                JST_TRACE("[REMOTE] Mouse: down (b='{}', x='{}', y='{}')", static_cast<int>(b), px, py);
                remoteInputStates[sessionId].mouseButtons.insert(b);
                pushInput(Render::InputQueue::Button{b, true});
                return;
            }
            if (act == "up") {
                JST_TRACE("[REMOTE] Mouse: up b={}, x={}, y={}", static_cast<int>(b), px, py);
                remoteInputStates[sessionId].mouseButtons.erase(b);
                pushInput(Render::InputQueue::Button{b, false});
                return;
            }
            if (act == "click" || act == "dblclick") {
                JST_TRACE("[REMOTE] Mouse: ignore (b='{}', x='{}', y='{}')", static_cast<int>(b), px, py);
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

        pushInput(Render::InputQueue::Wheel{{static_cast<F32>(dx), static_cast<F32>(dy)}});
        JST_TRACE("[REMOTE] Wheel: (dx='{}', dy='{}', mode='{}')", dx, dy, deltaMode);
        return;
    }

    if (kind == "clipboard") {
        if (!j.contains("text") || !j["text"].is_string()) {
            JST_WARN("[REMOTE] Rejected invalid clipboard text.");
            return;
        }
        const std::string text = j["text"].get<std::string>();
        if (text.size() > kMaxClipboardBytes) {
            JST_WARN("[REMOTE] Rejected clipboard text larger than {} bytes.", kMaxClipboardBytes);
            return;
        }
        if (viewport) {
            viewport->setClipboardText(text);
            sentClipboardRevision = viewport->clipboardRevision();
        }
        return;
    }

    if (kind == "text") {
        if (!j.contains("text") || !j["text"].is_string()) {
            JST_WARN("[REMOTE] Rejected invalid text input.");
            return;
        }
        const std::string text = j["text"].get<std::string>();
        if (text.empty() || text.size() > 256) {
            JST_WARN("[REMOTE] Rejected invalid text input size.");
            return;
        }
        for (const char32_t codepoint : Render::InputQueue::DecodeUtf8(text)) {
            pushInput(Render::InputQueue::Text{codepoint});
        }
        return;
    }

    const bool alt   = j.value("altKey",   false);
    const bool ctrl  = j.value("ctrlKey",  false);
    const bool shift = j.value("shiftKey", false);
    const bool meta  = j.value("metaKey",  false);

    if (kind == "keyboard") {
        const std::string action = j.value("action", "");
        if (action != "down" && action != "up") {
            JST_WARN("[REMOTE] Rejected invalid keyboard action '{}'.", action);
            return;
        }
        const std::string code = j.value("code", "");
        const std::string key = j.value("key", "");
        const bool pressed = (action == "down");

        auto& state = remoteInputStates[sessionId];
        state.alt = alt;
        state.ctrl = ctrl;
        state.shift = shift;
        state.meta = meta;

        const KeyCode k = mapKey(code, key);
        if (k != KeyCode::Unknown) {
            if (pressed) state.keys.insert(k);
            else state.keys.erase(k);
        }
        synchronizeKeyboardState();

        return;
    }

    JST_TRACE("[REMOTE] Unknown control (kind='{}').", kind);
}

void Instance::Remote::Impl::pushInput(Render::InputQueue::Event event) {
    if (!viewport) {
        return;
    }
    std::visit([](auto& typed) {
        if constexpr (requires { typed.remote; }) {
            typed.remote = true;
        }
    }, event);
    viewport->input().push(std::move(event));
}

Extent2D<F32> Instance::Remote::Impl::framebufferScale() const {
    return viewport ? viewport->metrics(1.0f).framebufferScale : Extent2D<F32>{1.0f, 1.0f};
}

void Instance::Remote::Impl::enqueueInput(std::string sessionId, nlohmann::json payload) {
    std::lock_guard<std::mutex> lock(inputMutex);
    if (inputSessionOrder.empty() || inputSessionOrder.front() != sessionId) {
        return;
    }
    inputQueue.push_back({std::move(sessionId), std::move(payload), false});
}

void Instance::Remote::Impl::enqueueInputReset(const std::string& sessionId) {
    std::lock_guard<std::mutex> lock(inputMutex);
    inputQueue.push_back({sessionId, {}, true});
}

void Instance::Remote::Impl::resetInput(const std::string& sessionId) {
    const auto it = remoteInputStates.find(sessionId);
    if (it == remoteInputStates.end()) {
        return;
    }
    for (const MouseButton button : it->second.mouseButtons) {
        pushInput(Render::InputQueue::Button{button, false});
    }
    remoteInputStates.erase(it);
    synchronizeKeyboardState();
}

void Instance::Remote::Impl::synchronizeKeyboardState() {
    std::unordered_set<KeyCode> keys;
    bool alt = false;
    bool ctrl = false;
    bool shift = false;
    bool meta = false;
    for (const auto& [sessionId, state] : remoteInputStates) {
        (void)sessionId;
        keys.insert(state.keys.begin(), state.keys.end());
        alt |= state.alt;
        ctrl |= state.ctrl;
        shift |= state.shift;
        meta |= state.meta;
    }

    pushInput(Render::InputQueue::Modifiers{KeyModifiers{
        .control = ctrl,
        .shift = shift,
        .alt = alt,
        .super = meta,
    }});

    for (const KeyCode key : appliedRemoteKeys) {
        if (!keys.contains(key)) pushInput(Render::InputQueue::Key{key, false});
    }
    for (const KeyCode key : keys) {
        if (!appliedRemoteKeys.contains(key)) pushInput(Render::InputQueue::Key{key, true});
    }
    appliedRemoteKeys = std::move(keys);
}


void Instance::Remote::Impl::handleControlMessage(const std::string& sessionId, std::string_view payload) {
    if (payload.empty() || payload.size() > kMaxClipboardMessageBytes) {
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

    if (j["kind"] != "clipboard" && payload.size() > kMaxControlBytes) {
        JST_WARN("[REMOTE] Rejected invalid control message size.");
        return;
    }

    if (j["kind"] == "viewport") {
        recordViewport(sessionId, j);
        return;
    }

    if (j["kind"] == "clipboard") {
        receiveClipboard(sessionId, j);
        return;
    }

    try {
        enqueueInput(sessionId, std::move(j));
    } catch (const std::exception& e) {
        JST_WARN("[REMOTE] Rejected malformed control message: {}", e.what());
    }
}

void Instance::Remote::Impl::receiveClipboard(const std::string& sessionId, const nlohmann::json& j) {
    const bool chunked = j.contains("id") || j.contains("index") || j.contains("count");
    const auto isCount = [&](const char* key) {
        return j.contains(key) && j[key].is_number_unsigned();
    };
    const bool valid = j.contains("text") && j["text"].is_string() &&
                       (!chunked || (isCount("id") && isCount("index") && isCount("count") &&
                                     j["text"].get_ref<const std::string&>().size() <= kClipboardChunkBytes));

    const U64 count = valid && chunked ? j["count"].get<U64>() : 1;
    const U64 index = valid && chunked ? j["index"].get<U64>() : 0;
    const U64 id = valid && chunked ? j["id"].get<U64>() : 0;

    std::string text;
    {
        std::lock_guard<std::mutex> lock(inputMutex);
        if (inputSessionOrder.empty() || inputSessionOrder.front() != sessionId) {
            clipboardTransfers.erase(sessionId);
            return;
        }
        if (!valid || count == 0 || count > kMaxClipboardChunks || index >= count) {
            clipboardTransfers.erase(sessionId);
            JST_WARN("[REMOTE] Rejected invalid clipboard chunk.");
            return;
        }

        auto& transfer = clipboardTransfers[sessionId];
        if (index == 0) {
            transfer = {.id = id, .count = count};
        } else if (transfer.id != id || transfer.count != count || transfer.next != index) {
            clipboardTransfers.erase(sessionId);
            JST_WARN("[REMOTE] Dropped an incomplete clipboard transfer.");
            return;
        }

        transfer.text += j["text"].get<std::string>();
        transfer.next = index + 1;
        if (transfer.text.size() > kMaxClipboardBytes) {
            clipboardTransfers.erase(sessionId);
            JST_WARN("[REMOTE] Rejected clipboard text larger than {} bytes.", kMaxClipboardBytes);
            return;
        }
        if (transfer.next < transfer.count) {
            return;
        }

        text = std::move(transfer.text);
        clipboardTransfers.erase(sessionId);
    }

    enqueueInput(sessionId, nlohmann::json{{"kind", "clipboard"}, {"text", std::move(text)}});
}

void Instance::Remote::Impl::recordViewport(const std::string& sessionId, const nlohmann::json& j) {
    constexpr F64 kMaxWidth = 3840.0;
    constexpr F64 kMaxHeight = 2160.0;
    constexpr F64 kMinSide = 240.0;
    constexpr F64 kMinPixels = 16.0;

    const F64 width = j.value("width", 0.0);
    const F64 height = j.value("height", 0.0);
    const F64 scale = std::clamp(j.value("scale", 1.0), 0.5, 4.0);
    if (!std::isfinite(width) || !std::isfinite(height) || width < 1.0 || height < 1.0 ||
        width > 65536.0 || height > 65536.0) {
        JST_WARN("[REMOTE] Rejected invalid viewport size.");
        return;
    }

    const F64 pixelsX = width * scale;
    const F64 pixelsY = height * scale;
    const F64 fit = std::min({1.0, kMaxWidth / pixelsX, kMaxHeight / pixelsY});
    const F64 grow = std::max(1.0, kMinSide / (std::min(pixelsX, pixelsY) * fit));
    const F64 factor = fit * std::min(grow, std::min(kMaxWidth / (pixelsX * fit), kMaxHeight / (pixelsY * fit)));

    const auto even = [](F64 value) {
        return static_cast<U64>(value) & ~static_cast<U64>(1);
    };

    std::lock_guard<std::mutex> lock(inputMutex);
    viewportRequests[sessionId] = {
        .size = {even(std::max(kMinPixels, pixelsX * factor)), even(std::max(kMinPixels, pixelsY * factor))},
        .scale = static_cast<F32>(scale * factor),
    };
}

void Instance::Remote::Impl::applyViewport() {
    if (!config.video.matchViewer) {
        return;
    }

    ViewportRequest request;
    {
        std::lock_guard<std::mutex> lock(inputMutex);
        std::erase_if(viewportRequests, [&](const auto& entry) {
            return std::ranges::find(inputSessionOrder, entry.first) == inputSessionOrder.end();
        });
        if (inputSessionOrder.empty()) {
            return;
        }
        const auto it = viewportRequests.find(inputSessionOrder.front());
        if (it == viewportRequests.end() || it->second == appliedViewport) {
            return;
        }
        request = it->second;
        appliedViewport = request;
    }

    if (viewport) {
        (void)viewport->resize(request.size, request.scale);
    }
}

void Instance::Remote::Impl::syncClipboard() {
    if (!viewport) {
        return;
    }

    const U64 revision = viewport->clipboardRevision();
    std::string controller;
    {
        std::lock_guard<std::mutex> lock(inputMutex);
        if (revision == sentClipboardRevision) {
            return;
        }
        if (inputSessionOrder.empty()) {
            sentClipboardRevision = revision;
            return;
        }
        controller = inputSessionOrder.front();
    }

    std::shared_ptr<rtc::DataChannel> channel;
    if (const auto session = findSession(controller)) {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        channel = session->control;
    }
    if (!channel || !channel->isOpen()) {
        return;
    }

    const auto change = viewport->clipboardChange();
    const auto markSent = [&] {
        std::lock_guard<std::mutex> lock(inputMutex);
        sentClipboardRevision = change.revision;
    };

    if (change.text.size() > kMaxClipboardBytes) {
        JST_WARN("[REMOTE] Clipboard text is too large to send to the viewer.");
        markSent();
        return;
    }

    try {
        const auto chunks = SplitUtf8(change.text, kClipboardChunkBytes);
        const U64 limit = std::min<U64>(channel->maxMessageSize(), kMaxClipboardMessageBytes);
        std::vector<std::string> messages;
        messages.reserve(chunks.size());
        for (U64 index = 0; index < chunks.size(); ++index) {
            messages.push_back(nlohmann::json{
                {"kind", "clipboard"},
                {"id", change.revision},
                {"index", index},
                {"count", chunks.size()},
                {"text", chunks[index]},
            }.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
            if (messages.back().size() > limit) {
                JST_WARN("[REMOTE] Clipboard text is too large to send to the viewer.");
                markSent();
                return;
            }
        }
        for (const auto& message : messages) {
            channel->send(message);
        }
        if (channel->isOpen()) {
            markSent();
        }
    } catch (const std::exception& e) {
        JST_WARN("[REMOTE] Failed to send clipboard text to the viewer: {}", e.what());
    }
}

void Instance::Remote::Impl::bindControlChannel(const std::string& sessionId,
                                                const std::shared_ptr<rtc::DataChannel>& channel) {
    const auto session = findSession(sessionId);
    if (!session) {
        channel->close();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        if (session->control) {
            JST_INFO("[REMOTE] Dropping duplicate control data channel.");
            channel->close();
            return;
        }
        session->control = channel;
    }

    channel->onClosed([this, sessionId]() {
        enqueueInputReset(sessionId);
    });
    channel->onMessage([](rtc::binary) {}, [this, sessionId](rtc::string text) {
        handleControlMessage(sessionId, text);
    });
}

Result Instance::Remote::Impl::openEncoder(std::unique_ptr<Encoder>& opened) {
    const auto open = [&]() {
        Encoder::Config encoderConfig = {
            .codec = config.video.codec,
            .encoder = encoderType,
            .device = inputMemoryDevice_,
            .size = streamSize(),
            .framerate = config.video.framerate,
        };

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        if (inputMemoryDevice_ == DeviceType::Vulkan) {
            encoderConfig.adapter = Viewport::FrameCaptureVulkan::Device();
        }
#endif

        opened = std::make_unique<Encoder>();
        return opened->create(encoderConfig);
    };

    Result result = open();
    for (const auto& fallback : encoderFallbacks) {
        if (result == Result::SUCCESS) {
            break;
        }

        inputMemoryDevice_ = fallback.device;
        encoderType = fallback.encoder;
        JST_WARN("[REMOTE] Falling back to {} encoding with {} memory.", fallback.encoder,
                                                                         GetDevicePrettyName(fallback.device));
        result = open();
    }

    return result;
}

Result Instance::Remote::Impl::startStream() {
    JST_DEBUG("[REMOTE] Starting stream.");

    std::lock_guard<std::mutex> lifecycle(lifecycleMutex);

    std::unique_ptr<Encoder> newEncoder;
    JST_CHECK(openEncoder(newEncoder));

    initialFrameTime = std::chrono::steady_clock::now();
    lastKeyframeTime = initialFrameTime;

    {
        std::lock_guard<std::mutex> lock(telemetry.mutex);
        telemetry.samples.clear();
        telemetry.keyframeBytes = 0;
        telemetry.format = newEncoder->format();
    }

    {
        std::lock_guard<std::mutex> lock(streamMutex);
        encoder = std::move(newEncoder);
        streaming = true;
    }

    requestKeyframe();

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::stopStream() {
    JST_DEBUG("[REMOTE] Stopping stream.");

    destroyAllWebRtcSessions();

    std::lock_guard<std::mutex> lifecycle(lifecycleMutex);
    std::unique_ptr<Encoder> oldEncoder;
    {
        std::lock_guard<std::mutex> lock(streamMutex);
        streaming = false;
        oldEncoder = std::move(encoder);
    }

    return Result::SUCCESS;
}

std::shared_ptr<Instance::Remote::Impl::WebRtcSession> Instance::Remote::Impl::findSession(const std::string& sessionId) {
    std::lock_guard<std::mutex> lock(sessionsMutex);
    const auto it = sessions.find(sessionId);
    return it == sessions.end() ? nullptr : it->second;
}

Result Instance::Remote::Impl::createWebRtcSession(const std::string& sessionId, const std::string& peerId) {
    if (findSession(sessionId)) {
        return Result::SUCCESS;
    }

    if (!streaming) {
        JST_ERROR("[REMOTE] Can't create WebRTC session without an active stream.");
        return Result::ERROR;
    }

    auto session = std::make_shared<WebRtcSession>();
    session->sessionId = sessionId;
    session->peerId = peerId;
    session->keyframe = std::make_shared<WebRtcSession::Keyframe>();
    session->keyframe->impl = this;

    rtc::Configuration configuration;
    configuration.disableAutoNegotiation = true;
    if (!config.media.address.empty()) {
        configuration.bindAddress = config.media.address;
    }
    if (config.media.minPort != 0) {
        configuration.portRangeBegin = config.media.minPort;
        configuration.portRangeEnd = config.media.maxPort;
    }

    try {
        session->peer = std::make_shared<rtc::PeerConnection>(configuration);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Failed to create WebRTC peer: {}", e.what());
        return Result::ERROR;
    }

    session->peer->onLocalDescription([this, sessionId](rtc::Description description) {
        sendSessionDescription(sessionId, description);
    });
    session->peer->onLocalCandidate([this, sessionId](rtc::Candidate candidate) {
        sendIceCandidate(sessionId, candidate);
    });
    session->peer->onStateChange([this](rtc::PeerConnection::State state) {
        if (state == rtc::PeerConnection::State::Connected) {
            requestKeyframe();
        }
    });
    session->peer->onDataChannel([this, sessionId](std::shared_ptr<rtc::DataChannel> channel) {
        if (channel->label() == "control") {
            JST_INFO("[REMOTE] Control data channel opened by client.");
            bindControlChannel(sessionId, channel);
            return;
        }
        JST_INFO("[REMOTE] Dropping data channel opened from client (label='{}').", channel->label());
        channel->close();
    });

    if (createVideoTrack(*session) != Result::SUCCESS) {
        closeWebRtcSession(*session);
        return Result::ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        sessions[sessionId] = session;

        std::lock_guard<std::mutex> inputLock(inputMutex);
        inputSessionOrder.push_back(sessionId);
    }
    requestKeyframe();

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::createVideoTrack(WebRtcSession& session) {
    U32 ssrc = 0;
    while (ssrc == 0) {
        ssrc = std::random_device{}();
    }

    const std::string parameters = Encoder::Parameters({
        .codec = config.video.codec,
        .size = streamSize(),
        .framerate = config.video.framerate,
    });

    rtc::Description::Video media(kVideoMid, rtc::Description::Direction::SendOnly);
    switch (config.video.codec) {
        case CodecType::H265:
            media.addH265Codec(kPayloadType, parameters);
            break;
        case CodecType::AV1:
            media.addAV1Codec(kPayloadType);
            break;
        case CodecType::H264:
        default:
            media.addH264Codec(kPayloadType, parameters);
            break;
    }
    media.addSSRC(ssrc, kCname, kCname, kVideoMid);
    media.addExtMap({kPlayoutDelayExtension, kPlayoutDelayUri});

    auto rtpConfig = std::make_shared<rtc::RtpPacketizationConfig>(ssrc,
                                                                    kCname,
                                                                    kPayloadType,
                                                                    rtc::RtpPacketizer::VideoClockRate);
    rtpConfig->playoutDelayId = kPlayoutDelayExtension;
    rtpConfig->playoutDelayMin = 0;
    rtpConfig->playoutDelayMax = kPlayoutDelayMaximum;

    std::shared_ptr<rtc::RtpPacketizer> packetizer;
    switch (config.video.codec) {
        case CodecType::H265:
            packetizer = std::make_shared<rtc::H265RtpPacketizer>(rtc::NalUnit::Separator::StartSequence,
                                                                  rtpConfig,
                                                                  kMaximumFragmentSize);
            break;
        case CodecType::AV1:
            packetizer = std::make_shared<rtc::AV1RtpPacketizer>(rtc::AV1RtpPacketizer::Packetization::TemporalUnit,
                                                                 rtpConfig,
                                                                 kMaximumFragmentSize);
            break;
        case CodecType::H264:
        default:
            packetizer = std::make_shared<rtc::H264RtpPacketizer>(rtc::NalUnit::Separator::StartSequence,
                                                                  rtpConfig,
                                                                  kMaximumFragmentSize);
            break;
    }

    packetizer->addToChain(std::make_shared<rtc::RtcpSrReporter>(rtpConfig));
    packetizer->addToChain(std::make_shared<rtc::RtcpNackResponder>(kNackHistory));
    packetizer->addToChain(std::make_shared<rtc::PliHandler>([keyframe = session.keyframe]() {
        std::lock_guard<std::mutex> lock(keyframe->mutex);
        if (keyframe->impl) {
            keyframe->impl->requestKeyframe();
        }
    }));

    try {
        session.track = session.peer->addTrack(std::move(media));
        session.track->setMediaHandler(packetizer);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Failed to add WebRTC video track: {}", e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

void Instance::Remote::Impl::closeWebRtcSession(WebRtcSession& session) {
    {
        std::lock_guard<std::mutex> lock(session.keyframe->mutex);
        session.keyframe->impl = nullptr;
    }

    session.peer->resetCallbacks();

    std::shared_ptr<rtc::DataChannel> control;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        control = std::move(session.control);
    }
    if (control) {
        control->resetCallbacks();
    }

    try {
        session.peer->close();
    } catch (const std::exception& e) {
        JST_DEBUG("[REMOTE] Failed to close WebRTC peer: {}", e.what());
    }
}

void Instance::Remote::Impl::destroyWebRtcSession(const std::string& sessionId) {
    std::shared_ptr<WebRtcSession> session;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        auto it = sessions.find(sessionId);
        if (it == sessions.end()) {
            return;
        }

        session = std::move(it->second);
        sessions.erase(it);

        std::lock_guard<std::mutex> inputLock(inputMutex);
        std::erase(inputSessionOrder, sessionId);
        clipboardTransfers.erase(sessionId);
        inputQueue.push_back({sessionId, {}, true});
    }

    JST_INFO("[REMOTE] Destroying WebRTC session '{}' with peer '{}'.", session->sessionId, session->peerId);

    closeWebRtcSession(*session);
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

    const auto session = findSession(sessionId);
    if (!session) {
        return;
    }

    try {
        bindControlChannel(sessionId, session->peer->createDataChannel("control"));
        session->peer->setLocalDescription(rtc::Description::Type::Offer);
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Failed to create WebRTC offer: {}", e.what());
        destroyWebRtcSession(sessionId);
        (void)sendSignallerMessage({{"type", "endSession"}, {"sessionId", sessionId}});
    }
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

        if (ice.contains("sdpMLineIndex") && !ice["sdpMLineIndex"].is_null()) {
            if (!ice["sdpMLineIndex"].is_number_unsigned() ||
                ice["sdpMLineIndex"].get<uint64_t>() > 65535) {
                JST_ERROR("[REMOTE] Peer message has an invalid ICE media index.");
                return;
            }
        }

        const std::string candidate = ice["candidate"].get<std::string>();
        std::string mid;
        if (ice.contains("sdpMid") && ice["sdpMid"].is_string() &&
            ice["sdpMid"].get_ref<const std::string&>().size() <= 64) {
            mid = ice["sdpMid"].get<std::string>();
        }

        if (!candidate.empty()) {
            const auto session = findSession(sessionId);
            if (session) {
                addRemoteCandidate(*session, candidate, mid);
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
                                                    const rtc::Description& description) {
    if (!findSession(sessionId)) {
        return;
    }

    const std::string sdp(description);
    if (sdp.empty()) {
        JST_ERROR("[REMOTE] Can't send empty WebRTC session description.");
        return;
    }

    nlohmann::json msg = {
        {"type", "peer"},
        {"sessionId", sessionId},
        {"sdp", {{"type", description.typeString()}, {"sdp", sdp}}},
    };

    if (!sendSignallerMessage(msg)) {
        JST_ERROR("[REMOTE] Failed to send WebRTC session description.");
    }
}

void Instance::Remote::Impl::sendIceCandidate(const std::string& sessionId,
                                              const rtc::Candidate& candidate) {
    const std::string line = candidate.candidate();
    if (sessionId.empty() || line.empty()) {
        return;
    }

    if (!findSession(sessionId)) {
        return;
    }

    nlohmann::json msg = {
        {"type", "peer"},
        {"sessionId", sessionId},
        {"ice", {{"candidate", line}, {"sdpMid", candidate.mid()}, {"sdpMLineIndex", 0}}},
    };

    if (!sendSignallerMessage(msg)) {
        JST_ERROR("[REMOTE] Failed to send WebRTC ICE candidate.");
    }
}

void Instance::Remote::Impl::addRemoteCandidate(WebRtcSession& session,
                                                const std::string& candidate,
                                                const std::string& mid) {
    const std::string target = mid.empty() ? kVideoMid : mid;
    {
        std::lock_guard<std::mutex> lock(session.candidatesMutex);
        if (session.received >= kMaximumCandidates) {
            JST_WARN("[REMOTE] Dropping WebRTC candidate beyond the session limit.");
            return;
        }
        ++session.received;

        if (!session.described) {
            session.candidates.emplace_back(candidate, target);
            return;
        }
    }

    try {
        session.peer->addRemoteCandidate(rtc::Candidate(candidate, target));
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Failed to add WebRTC candidate: {}", e.what());
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

    const auto session = findSession(sessionId);
    if (!session) {
        JST_ERROR("[REMOTE] WebRTC session '{}' does not exist.", sessionId);
        return Result::ERROR;
    }

    try {
        session->peer->setRemoteDescription(rtc::Description(sdpText, type));
        if (type == "offer") {
            session->peer->setLocalDescription(rtc::Description::Type::Answer);
        }
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Failed to apply WebRTC {} for session '{}': {}", type, sessionId, e.what());
        return Result::ERROR;
    }

    std::vector<std::pair<std::string, std::string>> queued;
    {
        std::lock_guard<std::mutex> lock(session->candidatesMutex);
        session->described = true;
        queued.swap(session->candidates);
    }
    for (const auto& [candidate, mid] : queued) {
        try {
            session->peer->addRemoteCandidate(rtc::Candidate(candidate, mid));
        } catch (const std::exception& e) {
            JST_ERROR("[REMOTE] Failed to add WebRTC candidate: {}", e.what());
        }
    }

    return Result::SUCCESS;
}

//
// Frame Submission
//

Result Instance::Remote::Impl::pushNewFrame(const void* data) {
    std::vector<std::shared_ptr<WebRtcSession>> targets;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        if (sessions.empty()) {
            return Result::SUCCESS;
        }
        for (const auto& [sessionId, session] : sessions) {
            (void)sessionId;
            targets.push_back(session);
        }
    }

    packets.clear();
    {
        std::lock_guard<std::mutex> lock(streamMutex);

        if (!streaming || !encoder) {
            return Result::SUCCESS;
        }

        const auto currentFrameTime = std::chrono::steady_clock::now();
        const auto elapsedSinceStart = std::chrono::duration_cast<std::chrono::microseconds>(currentFrameTime -
                                                                                            initialFrameTime);
        const auto elapsedSinceLastKeyframe = std::chrono::duration_cast<std::chrono::seconds>(currentFrameTime -
                                                                                               lastKeyframeTime);

        bool keyframe = forceKeyframe.exchange(false);
        if (elapsedSinceLastKeyframe.count() > 1 || keyframe) {
            keyframe = true;
            lastKeyframeTime = currentFrameTime;
        }

        const auto encodeStart = std::chrono::steady_clock::now();
        const Result encoded = encoder->encode(data, elapsedSinceStart, keyframe, packets);
        const auto encodeEnd = std::chrono::steady_clock::now();
        const std::chrono::duration<F32, std::milli> encodeTime = encodeEnd - encodeStart;
        if (encoded == Result::SUCCESS) {
            record(encodeEnd, encodeTime.count(), packets);
        } else {
            record(encodeEnd, encodeTime.count(), {});
        }
        JST_CHECK(encoded);
    }

    for (const auto& packet : packets) {
        for (const auto& session : targets) {
            if (!session->track->isOpen()) {
                continue;
            }
            try {
                const std::chrono::duration<double, std::micro> time(static_cast<double>(packet.timestamp.count()));
                session->track->sendFrame(reinterpret_cast<const rtc::byte*>(packet.data.data()),
                                          packet.data.size(),
                                          rtc::FrameInfo(time));
            } catch (const std::exception& e) {
                JST_DEBUG("[REMOTE] Failed to send WebRTC media frame: {}", e.what());
            }
        }
    }

    return Result::SUCCESS;
}

void Instance::Remote::Impl::record(std::chrono::steady_clock::time_point time,
                                    F32 encode,
                                    const std::vector<Encoder::Packet>& output) {
    U64 bytes = 0;
    U64 keyframeBytes = 0;
    for (const auto& packet : output) {
        bytes += packet.data.size();
        if (packet.keyframe) {
            keyframeBytes += packet.data.size();
        }
    }

    std::lock_guard<std::mutex> lock(telemetry.mutex);
    telemetry.samples.push_back({time, encode, bytes});
    while (telemetry.samples.size() > kTelemetryLimit ||
           time - telemetry.samples.front().time > std::chrono::seconds(1)) {
        telemetry.samples.pop_front();
    }
    if (keyframeBytes > 0) {
        telemetry.keyframeBytes = keyframeBytes;
    }
}

Instance::Remote::Stats Instance::Remote::Impl::stats() {
    Stats stats;
    if (!streaming) {
        return stats;
    }

    stats.encoder = encoderType;
    stats.codec = config.video.codec;
    stats.memory = inputMemoryDevice_;
    stats.size = streamSize();
    stats.targetRate = static_cast<F32>(config.video.framerate);

    std::vector<std::shared_ptr<WebRtcSession>> active;
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);
        for (const auto& [sessionId, session] : sessions) {
            (void)sessionId;
            active.push_back(session);
        }
    }
    for (const auto& session : active) {
        if (!session->track || !session->track->isOpen()) {
            continue;
        }
        if (const auto roundTrip = session->peer->rtt()) {
            stats.roundTrip = std::max(stats.roundTrip.value_or(0.0f), static_cast<F32>(roundTrip->count()));
        }
    }

    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(telemetry.mutex);
    stats.format = telemetry.format;
    stats.keyframeBytes = telemetry.keyframeBytes;

    U64 frames = 0;
    U64 bytes = 0;
    U64 calls = 0;
    F32 encodeTotal = 0.0f;
    for (const auto& sample : telemetry.samples) {
        if (now - sample.time > std::chrono::seconds(1)) {
            continue;
        }
        ++calls;
        encodeTotal += sample.encode;
        if (sample.bytes > 0) {
            ++frames;
            bytes += sample.bytes;
        }
    }

    stats.frameRate = static_cast<F32>(frames);
    stats.bitrate = static_cast<F32>(bytes * 8);
    stats.encodeAverage = calls > 0 ? encodeTotal / static_cast<F32>(calls) : 0.0f;

    return stats;
}

}  // namespace Jetstream
